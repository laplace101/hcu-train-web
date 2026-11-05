"""MiniMax family.

MiniMax's models are MoE decoders with a few family-specific details:

  * the expert count is written to `num_local_experts`, not to the
    `num_experts` / `n_routed_experts` the kernel reads;
  * the expert FFN width is `mlp_intermediate_size`, while
    `intermediate_size` is the *dense* MLP width — a different number;
  * the router uses a sigmoid gate with a routing bias;
  * there is a multi-token-prediction module (`num_mtp_modules` /
    `mtp_transformer_layers` / `use_mtp`);
  * the released weights are FP8-quantised.

The attention layout differs between the generations, and this is the part
that needs care:

  * **M2 and later** use **full multi-head attention across all layers**.
    The official implementation has a single `MiniMaxM2Attention` class and
    no linear-attention variant; the MiniMax-M2 paper states that M2
    "adopts full multi-head attention across all layers, departing from the
    hybrid design used in MiniMax-Text-01". The `attn_type_list` field is
    still present in the released config but carries the default
    `[1] * num_hidden_layers` and the official modelling code never reads
    it. So for these models *every* layer is a full-attention layer.

  * **Text-01 and M1** interleave Lightning Attention with full attention,
    and declare the layout in `attn_type_list`: `1` for a full-attention
    layer, `0` for a Lightning (linear-attention) layer. Text-01's list
    has a full-attention layer every eighth layer.

The kernel's linear-attention spec is Gated DeltaNet, not Lightning
Attention. The two are different operators — Lightning has no delta-rule
state update — so mapping Lightning onto it is an approximation, not an
equivalence. The adapter therefore uses it only where the config actually
asks for a linear-attention layer, and it says so in the compatibility
check rather than presenting the result as exact.

The build reuses the kernel's own specs, so the numbers are the kernel's
numbers.
"""

from __future__ import annotations

from typing import Any, Mapping

from hcu_train_simulator.models.base import (
    ModelAdapter,
    NormalizedModelInput,
    with_rmsnorm_defaults,
)

MODEL_TYPES = (
    "minimax_m2",
    "minimax_m1",
    "minimax_text_01",
    "minimax_text01",
    "minimax",
)

MODEL_TYPE_PREFIXES = ("minimax",)

ARCHITECTURE_PREFIXES = ("MiniMax",)

#: The value `attn_type_list` uses for a full-attention layer.
FULL_ATTENTION = 1

#: The value it uses for a Lightning (linear-attention) layer.
LINEAR_ATTENTION = 0

#: The config fields that carry the expert count.
EXPERT_COUNT_FIELDS = ("num_experts", "n_routed_experts", "num_local_experts")

#: The config fields that carry the expert FFN width.
MOE_FFN_FIELDS = ("moe_intermediate_size", "moe_ffn_hidden_size")

#: The config fields that carry the number of experts routed per token.
TOPK_FIELDS = ("num_experts_per_tok", "num_experts_per_token", "topk")

#: The config fields that carry the MTP module size.
MTP_FIELDS = ("num_mtp_modules", "mtp_num_layers", "num_nextn_predict_layers")


class MiniMaxFamilyAdapter(ModelAdapter):
    """MiniMax MoE decoders: sigmoid-gated experts, optional MTP."""

    name = "minimax_family"
    model_types = MODEL_TYPES
    model_type_prefixes = MODEL_TYPE_PREFIXES
    architecture_prefixes = ARCHITECTURE_PREFIXES

    def normalize_config(self, raw_config: Mapping[str, Any]) -> NormalizedModelInput:
        normalised = self._normalise(dict(raw_config))

        # The kernel's model config keeps only the fields it names, so the
        # attention layout has to be handed over in the form the kernel
        # reads: `layer_types`. This is the same normalisation the kernel's
        # own `qwen3_5` adapter performs, just sourced from
        # `attn_type_list` instead.
        if not normalised.get("layer_types"):
            attn_list = normalised.get("attn_type_list")
            if attn_list:
                normalised["layer_types"] = tuple(
                    "full_attention" if int(value) == FULL_ATTENTION
                    else "linear_attention"
                    for value in attn_list
                )

        return NormalizedModelInput(
            language=with_rmsnorm_defaults(normalised)
        )

    def build_spec(self, simulation_config):
        from hcu_train_simulator.modeling.te_spec import (
            build_transformer_engine_model_spec,
        )

        transformer = simulation_config.transformer
        layer_types = self._layer_types(transformer)

        layer_attention_specs = None
        mtp_attention_spec = None
        if any(kind == "linear_attention" for kind in layer_types):
            from hcu_train_simulator.models.qwen3_5.spec import (
                build_gated_attention_spec,
                build_gated_delta_net_spec,
            )

            layer_attention_specs = tuple(
                build_gated_delta_net_spec()
                if kind == "linear_attention"
                else build_gated_attention_spec()
                for kind in layer_types
            )
            mtp_attention_spec = build_gated_attention_spec()

        mtp_model_spec = None
        if transformer.mtp_num_layers:
            from hcu_train_simulator.modeling.te_spec import (
                build_multi_token_predictor_spec,
            )

            mtp_model_spec = build_multi_token_predictor_spec()

        return build_transformer_engine_model_spec(
            transformer,
            simulation_config.model,
            qk_layernorm=True,
            layer_attention_specs=layer_attention_specs,
            mtp_attention_spec=mtp_attention_spec,
            mtp_model_spec=mtp_model_spec,
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )

    def validate(self, simulation_config) -> None:
        transformer = simulation_config.transformer
        layer_types = self._layer_types(transformer)
        if len(layer_types) != transformer.num_layers:
            raise AssertionError(
                "the attention layout must contain one entry per layer"
            )

    # -- internals ---------------------------------------------------------

    @classmethod
    def _normalise(cls, language: dict[str, Any]) -> dict[str, Any]:
        """Rename the fields the kernel reads, without inventing values."""
        normalised = dict(language)

        # Expert count → num_experts.
        if normalised.get("num_experts") is None:
            for field in EXPERT_COUNT_FIELDS:
                value = normalised.get(field)
                if value is not None:
                    normalised["num_experts"] = value
                    break

        # Expert FFN width.
        #
        # M2 and later state it as `mlp_intermediate_size`, because they
        # also have a separate dense MLP whose width is `intermediate_size`
        # — the two numbers differ (8192 against 1536) and substituting one
        # for the other would be wrong.
        #
        # M1 and Text-01 have no dense MLP alongside the experts, so their
        # `intermediate_size` *is* the expert width and is the right value
        # to use. Which of the two a config belongs to is decided by which
        # field it states, not by the model name.
        if not any(normalised.get(f) is not None for f in MOE_FFN_FIELDS):
            width = normalised.get("mlp_intermediate_size")
            if width is None:
                width = normalised.get("intermediate_size")
            if width is not None:
                normalised["moe_intermediate_size"] = width

        # Routed experts per token.
        if normalised.get("num_experts_per_tok") is None:
            for field in TOPK_FIELDS:
                value = normalised.get(field)
                if value is not None:
                    normalised["num_experts_per_tok"] = value
                    break

        # The MTP module. `num_mtp_modules` is the number of prediction
        # modules; the kernel counts the layers it adds.
        if normalised.get("mtp_num_layers") is None:
            for field in MTP_FIELDS:
                value = normalised.get(field)
                if value is not None:
                    normalised["mtp_num_layers"] = value
                    break

        return normalised

    @staticmethod
    def _layer_types(transformer) -> tuple[str, ...]:
        """The per-layer attention kind.

        `normalize_config` has already translated `attn_type_list` into
        `layer_types`, so the transformer carries the layout the kernel
        understands: `1` marked a full-attention layer and `0` a Lightning
        (linear-attention) layer.

        A list that is entirely full attention produces no linear-attention
        layers at all — which is the case for MiniMax-M2, whose released
        config still carries the field with its default value while the
        model itself uses full attention everywhere.
        """
        layer_types = transformer.layer_types
        if layer_types:
            return tuple(layer_types)
        return tuple("full_attention" for _ in range(transformer.num_layers))
