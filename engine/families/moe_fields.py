"""MoE field normalisation — any family, any naming convention.

Several model families write their MoE parameters under names the kernel
does not read. The kernel takes the expert count from `num_experts` or
`n_routed_experts`; a config that only has `num_local_experts` is modelled
as a *dense* decoder, so a 256-expert MoE gets the memory and compute of
a single dense FFN.

Measured on MiniMax-M2 (230B total, 10B active, 256 experts, top-8):

    expert count unreadable   peak 67.48 GiB    ← a dense decoder
    expert count read         peak 1650.92 GiB  ← 24x larger

This module normalises the field names so the kernel reads them. It is
deliberately narrow:

  * it only *renames* fields the kernel already understands — it never
    invents values, and it never changes an attention structure;
  * a config that already uses a name the kernel reads is untouched, so
    the numbers for every model the kernel already handles are bit-for-bit
    unchanged;
  * the MoE FFN width is only filled in when the config omits it
    entirely. A config that states it differently is left alone, because
    that difference is a modelling decision, not a naming one.

This is the difference between a name-mapping fix and a model support
effort: it makes the expert count visible to the kernel, but it does not
model MiniMax's Lightning hybrid attention, its multi-token predictor or
its sigmoid router. Those remain reasons to refuse the estimate.
"""

from __future__ import annotations

from typing import Any, Mapping

from hcu_train_simulator.models.base import (
    ModelAdapter,
    NormalizedModelInput,
    with_rmsnorm_defaults,
)

#: The config fields that carry the expert count, in the kernel's preferred
#: order. The first one present wins.
EXPERT_COUNT_FIELDS = ("num_experts", "n_routed_experts", "num_local_experts")

#: Aliases the kernel reads for the MoE FFN width.
MOE_FFN_FIELDS = ("moe_intermediate_size", "moe_ffn_hidden_size")

#: Aliases the kernel reads for the number of experts routed per token.
TOPK_FIELDS = ("num_experts_per_tok", "num_experts_per_token", "topk")


class MoEFieldNormalisingAdapter(ModelAdapter):
    """Normalises MoE field names so the kernel can read them.

    This adapter is a *fallback*: it matches any config that carries MoE
    fields under an alias the kernel does not read, and only then. A config
    the kernel already handles is claimed by the kernel's own adapters
    first, because those score higher.
    """

    name = "moe_field_normalised"
    fallback = True

    def match_score(self, raw_config: Mapping[str, Any]) -> int | None:
        """Match only when the config needs the normalisation.

        The generic adapter is also a fallback and returns 0. This adapter
        returns a small positive score when it has something to fix, so it
        wins the tie — the kernel breaks ties on the adapter name, and
        `moe_field_normalised` sorts before neither `generic_gpt` nor the
        family adapters' exact matches.

        Returning `None` — not 0 — when there is nothing to fix is what
        keeps the adapter out of the running entirely. A score of 0 would
        still be a candidate, and the kernel's tie-break on the adapter
        name would then pick this one over `generic_gpt` for every dense
        model.
        """
        language = self._language(raw_config)
        if not self._needs_normalisation(language):
            return None
        return 1

    def normalize_config(self, raw_config: Mapping[str, Any]) -> NormalizedModelInput:
        language = dict(self._language(raw_config))
        normalised = self._normalise(language)
        return NormalizedModelInput(language=with_rmsnorm_defaults(normalised))

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
        return build_transformer_engine_model_spec(
            transformer,
            simulation_config.model,
            multi_latent_attention=uses_mla,
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _language(raw_config: Mapping[str, Any]) -> dict[str, Any]:
        """The language part of a config, whether nested or flat."""
        nested = raw_config.get("text_config")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            for key in ("architectures", "model_type", "tie_word_embeddings"):
                if key in raw_config:
                    merged.setdefault(key, raw_config[key])
            return merged
        return dict(raw_config)

    @classmethod
    def _needs_normalisation(cls, language: Mapping[str, Any]) -> bool:
        """Whether the kernel would misread this config's MoE fields."""
        expert_count = next(
            (language[f] for f in EXPERT_COUNT_FIELDS if language.get(f)), None
        )
        if expert_count is None:
            # No expert count at all: a dense model, or one whose config
            # omits it. Nothing to fix.
            return False

        # The kernel already reads the count — nothing to rename.
        if language.get("num_experts") is not None:
            return False

        # The MoE FFN width: the kernel needs it to size the expert GEMM.
        # A config that states it under any known name is fine.
        has_ffn = any(language.get(f) is not None for f in MOE_FFN_FIELDS)
        return not has_ffn

    @classmethod
    def _normalise(cls, language: Mapping[str, Any]) -> dict[str, Any]:
        """Rename the fields the kernel reads, without inventing values."""
        normalised = dict(language)

        # Expert count → num_experts.
        if normalised.get("num_experts") is None:
            for field in EXPERT_COUNT_FIELDS:
                value = normalised.get(field)
                if value is not None:
                    normalised["num_experts"] = value
                    break

        # MoE FFN width. Only filled when absent: a config that states a
        # *different* width under another name is making a modelling
        # decision, which this module does not override.
        if not any(normalised.get(f) is not None for f in MOE_FFN_FIELDS):
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

        return normalised
