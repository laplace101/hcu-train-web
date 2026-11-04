"""GLM family.

GLM-5 uses DSA (sparse index attention) with MLA-style low-rank KV, an
optional MoE FFN, and an optional multi-token predictor. The kernel's
`glm5` adapter models exactly that and matches on `model_type:
glm_moe_dsa`.

This module covers the family by capability: a config that carries the DSA
index fields is a GLM-family model regardless of what it calls itself.
That covers `glm5`, `glm5_2`, `glm5_3` and any later release that keeps the
same attention structure.

GLM-4 models are a different generation — standard attention, no DSA. They
have no `index_*` fields, so they do not match this adapter and continue to
fall through to the generic fallback, which is the correct treatment for
them: their attention is the standard one the generic adapter computes.

The build reuses the kernel's own DSA specs, so the numbers match the
kernel's `glm5` adapter exactly.
"""

from __future__ import annotations

from typing import Any, Mapping

from hcu_train_simulator.models.base import (
    ModelAdapter,
    NormalizedModelInput,
    with_rmsnorm_defaults,
)

MODEL_TYPES = (
    "glm_moe_dsa",
    "glm5",
    "glm5_moe",
    "glm5_2",
    "glm5_3",
)

MODEL_TYPE_PREFIXES = ("glm_moe_dsa", "glm5")

ARCHITECTURE_PREFIXES = ("GlmMoeDsa", "Glm5")


class GLMFamilyAdapter(ModelAdapter):
    """GLM-5 family: DSA index attention, MLA-style KV, optional MoE/MTP."""

    name = "glm_family"
    model_types = MODEL_TYPES
    model_type_prefixes = MODEL_TYPE_PREFIXES
    architecture_prefixes = ARCHITECTURE_PREFIXES

    def normalize_config(self, raw_config: Mapping[str, Any]) -> NormalizedModelInput:
        return NormalizedModelInput(language=with_rmsnorm_defaults(dict(raw_config)))

    def build_spec(self, simulation_config):
        from hcu_train_simulator.modeling.te_spec import (
            build_transformer_engine_model_spec,
        )
        from hcu_train_simulator.models.glm5.spec import build_dsa_attention_spec

        transformer = simulation_config.transformer
        indexer_types = transformer.indexer_types or ("full",) * transformer.num_layers
        attention_specs = tuple(
            build_dsa_attention_spec(include_indexer=indexer_type == "full")
            for indexer_type in indexer_types
        )

        mtp_model_spec = None
        if transformer.mtp_num_layers:
            from hcu_train_simulator.modeling.te_spec import (
                build_multi_token_predictor_spec,
            )

            mtp_model_spec = build_multi_token_predictor_spec()

        return build_transformer_engine_model_spec(
            transformer,
            simulation_config.model,
            layer_attention_specs=attention_specs,
            mtp_attention_spec=build_dsa_attention_spec(include_indexer=True),
            mtp_model_spec=mtp_model_spec,
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )

    def validate(self, simulation_config) -> None:
        transformer = simulation_config.transformer
        if transformer.index_head_dim <= 0:
            raise AssertionError("index_head_dim must be positive for a DSA model.")
        if transformer.moe_layer_freq <= 0:
            raise AssertionError("moe_layer_freq must be positive.")
