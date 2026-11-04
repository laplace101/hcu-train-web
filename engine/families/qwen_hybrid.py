"""Qwen hybrid-attention family.

Qwen's newer models interleave two kinds of attention:

  * **linear attention** — Gated DeltaNet, a recurrent layer whose cost
    scales with the sequence length rather than with its square;
  * **full attention** — the standard quadratic layer.

A model declares the layout in one of two ways:

  * `layer_types` — an explicit per-layer list, e.g.
    `["linear_attention", "linear_attention", "linear_attention",
    "full_attention", ...]`;
  * `full_attention_interval` + `linear_conv_kernel_dim` — a periodic
    layout, e.g. every 4th layer is full attention.

The kernel's own `qwen3_5` adapter handles the first form and derives the
second from `full_attention_interval`. But its `model_types` list names
`qwen3_5` and friends explicitly, so a model declaring `qwen3_next` — same
architecture, different name — falls to the `qwen3` adapter, which applies
standard attention to every layer.

This adapter covers the family by capability rather than by name: any
config that carries hybrid-attention markers gets the hybrid treatment,
whatever it calls itself.

The build is a thin extension of the kernel's own specs — the layer specs
come from `models.qwen3_5.spec` — so a config the kernel already handles
produces the kernel's numbers, not a second implementation of them.
"""

from __future__ import annotations

from typing import Any, Mapping

from hcu_train_simulator.models.base import (
    ModelAdapter,
    NormalizedModelInput,
    with_rmsnorm_defaults,
)

#: The config fields that mark a config as using hybrid attention.
HYBRID_MARKERS = (
    "layer_types",
    "full_attention_interval",
    "linear_conv_kernel_dim",
)

#: The `model_type` values this family owns. Prefix matching is used as
#: well, so a new member of the series is covered without a new entry.
MODEL_TYPES = (
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_text",
    "qwen3_5_moe_text",
    "qwen3_next",
    "qwen3_next_moe",
    "qwen3_5_next",
    "qwen3_5_next_moe",
)

MODEL_TYPE_PREFIXES = ("qwen3_5", "qwen3_next")

ARCHITECTURE_PREFIXES = ("Qwen3_5", "Qwen3Next", "Qwen3NextMoe")


class QwenHybridAdapter(ModelAdapter):
    """Hybrid-attention Qwen models: linear attention interleaved with full."""

    name = "qwen_hybrid"
    model_types = MODEL_TYPES
    model_type_prefixes = MODEL_TYPE_PREFIXES
    architecture_prefixes = ARCHITECTURE_PREFIXES

    def normalize_config(self, raw_config: Mapping[str, Any]) -> NormalizedModelInput:
        return NormalizedModelInput(language=with_rmsnorm_defaults(dict(raw_config)))

    def build_spec(self, simulation_config):
        from hcu_train_simulator.modeling.te_spec import (
            build_transformer_engine_model_spec,
        )
        from hcu_train_simulator.models.qwen3_5.spec import (
            build_gated_attention_spec,
            build_gated_delta_net_spec,
        )

        transformer = simulation_config.transformer
        layer_types = self._layer_types(transformer)
        attention_specs = tuple(
            build_gated_delta_net_spec()
            if layer_type == "linear_attention"
            else build_gated_attention_spec()
            for layer_type in layer_types
        )
        vision_spec = self._vision_spec(simulation_config)

        return build_transformer_engine_model_spec(
            transformer,
            simulation_config.model,
            qk_layernorm=True,
            layer_attention_specs=attention_specs,
            mtp_attention_spec=build_gated_attention_spec(),
            mtp_model_spec=self._mtp_spec(transformer),
            vision_model_spec=vision_spec,
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )

    def validate(self, simulation_config) -> None:
        transformer = simulation_config.transformer
        layer_types = self._layer_types(transformer)
        if len(layer_types) != transformer.num_layers:
            raise AssertionError(
                "hybrid attention layout must contain one entry per layer"
            )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _layer_types(transformer) -> tuple[str, ...]:
        """The per-layer attention kind.

        The explicit list wins. Otherwise the periodic layout is derived
        from `full_attention_interval`, exactly as the kernel's own
        `qwen3_5` adapter does.
        """
        layer_types = transformer.layer_types
        if layer_types:
            return tuple(layer_types)

        interval = transformer.full_attention_interval or 0
        if not interval:
            # No hybrid markers at all: every layer is full attention.
            return tuple("full_attention" for _ in range(transformer.num_layers))
        return tuple(
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(transformer.num_layers)
        )

    @staticmethod
    def _mtp_spec(transformer):
        if not transformer.mtp_num_layers:
            return None
        from hcu_train_simulator.modeling.te_spec import (
            build_multi_token_predictor_spec,
        )

        return build_multi_token_predictor_spec()

    @staticmethod
    def _vision_spec(simulation_config):
        if not simulation_config.vision.enabled:
            return None
        from hcu_train_simulator.models.common.qwen_vl.spec import (
            build_vision_model_spec,
        )

        return build_vision_model_spec(
            simulation_config.vision, model_type="qwen3_5_vision"
        )
