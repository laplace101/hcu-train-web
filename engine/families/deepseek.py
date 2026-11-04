"""DeepSeek family.

DeepSeek's models share one architecture: Multi-Latent Attention (MLA) with
a low-rank KV cache, an optional Mixture-of-Experts FFN, and an optional
multi-token predictor. The kernel's `deepseek_v3` adapter already models
that, and matches on the `deepseek` `model_type` prefix, so V2, V3 and R1
all resolve to it.

This module exists for two reasons:

  * the kernel's adapter only matches on a `model_type` prefix, so a
    DeepSeek model that declares something else (`deepseek_v4`,
    `deepseek_v31`) still matches — but a model that reaches the generic
    fallback would be computed with the standard dense-decoder formulas
    instead of MLA. This adapter keys on the *capability* (the MLA fields
    being present) rather than on the name;
  * the newer DeepSeek releases add DSA (sparse index attention), which
    the kernel only applies through the `glm5` adapter. This adapter
    detects the DSA fields and applies the same sparse-attention specs, so
    the whole series is covered by one module.

The build extends the kernel's own specs rather than reimplementing them,
so a config the kernel already handles produces the kernel's numbers.
"""

from __future__ import annotations

from typing import Any, Mapping

from hcu_train_simulator.models.base import (
    ModelAdapter,
    NormalizedModelInput,
    with_rmsnorm_defaults,
)

MODEL_TYPES = (
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v3_1",
    "deepseek_v3_2",
    "deepseek_v31",
    "deepseek_v32",
    "deepseek_v4",
    "deepseek_r1",
)

MODEL_TYPE_PREFIXES = ("deepseek",)

ARCHITECTURE_PREFIXES = ("Deepseek", "DeepSeek")


class DeepSeekFamilyAdapter(ModelAdapter):
    """DeepSeek models: MLA, optional MoE, optional DSA, optional MTP."""

    name = "deepseek_family"
    model_types = MODEL_TYPES
    model_type_prefixes = MODEL_TYPE_PREFIXES
    architecture_prefixes = ARCHITECTURE_PREFIXES

    def normalize_config(self, raw_config: Mapping[str, Any]) -> NormalizedModelInput:
        return NormalizedModelInput(language=with_rmsnorm_defaults(dict(raw_config)))

    def build_spec(self, simulation_config):
        from hcu_train_simulator.modeling.te_spec import (
            build_transformer_engine_model_spec,
        )

        transformer = simulation_config.transformer
        uses_mla = bool(
            transformer.kv_lora_rank
            and transformer.qk_nope_head_dim
            and transformer.qk_rope_head_dim
        )

        layer_attention_specs = None
        mtp_attention_spec = None
        if self._uses_dsa(transformer):
            from hcu_train_simulator.models.glm5.spec import build_dsa_attention_spec

            indexer_types = transformer.indexer_types or (
                "full",
            ) * transformer.num_layers
            layer_attention_specs = tuple(
                build_dsa_attention_spec(include_indexer=indexer_type == "full")
                for indexer_type in indexer_types
            )
            mtp_attention_spec = build_dsa_attention_spec(include_indexer=True)

        mtp_model_spec = None
        if transformer.mtp_num_layers:
            from hcu_train_simulator.modeling.te_spec import (
                build_multi_token_predictor_spec,
            )

            mtp_model_spec = build_multi_token_predictor_spec()

        return build_transformer_engine_model_spec(
            transformer,
            simulation_config.model,
            multi_latent_attention=uses_mla,
            layer_attention_specs=layer_attention_specs,
            mtp_attention_spec=mtp_attention_spec,
            mtp_model_spec=mtp_model_spec,
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )

    def validate(self, simulation_config) -> None:
        transformer = simulation_config.transformer
        if transformer.moe_layer_freq <= 0:
            raise AssertionError("moe_layer_freq must be positive.")
        if not 0 <= transformer.first_k_dense_replace <= transformer.num_layers:
            raise AssertionError(
                "first_k_dense_replace must be between zero and num_hidden_layers."
            )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _uses_dsa(transformer) -> bool:
        """Whether the config carries DSA (sparse index attention) fields."""
        return bool(
            transformer.index_head_dim
            and transformer.index_n_heads
            and transformer.index_topk
        )
