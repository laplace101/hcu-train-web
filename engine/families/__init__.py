"""Model family adapters (plan: one generic module per series).

The estimator kernel ships six adapters, each bound to one model family.
That is fine for the families the kernel knows, but a model that belongs to
a *known* family yet declares a slightly different `model_type` lands on
the wrong adapter — or on the generic fallback, which silently applies the
standard dense-decoder formulas.

Two examples, both measured on this machine:

  * Qwen3-Next declares `model_type: qwen3_next`. The kernel's `qwen3`
    adapter matches on the `qwen3` prefix, so it computes every layer with
    standard attention. Qwen3-Next is a *hybrid* model — it interleaves
    Gated DeltaNet layers with full-attention layers. Running the same
    config through the hybrid-aware adapter instead changes the iteration
    time by **+15.5%** and the throughput by **-13.4%**.

  * MiniMax writes its expert count to `num_local_experts`. The kernel
    reads `num_experts` / `n_routed_experts`, so a 256-expert MoE is
    modelled as a dense decoder: the peak memory comes out **24x** too
    small (67.5 GiB instead of 1651 GiB).

Both are the same class of problem: a family whose members differ in *how
they declare* an architecture the kernel already knows how to model.

The modules here fix that at the family level rather than per model. Each
one owns a whole series and derives the architecture from the config
fields, so a new member of the series is covered without a new adapter:

    QwenHybridFamily   qwen3.5 / qwen3-next / qwen3.5-next  (hybrid attention)
    DeepSeekFamily     deepseek v2 / v3 / r1                (MLA + MoE)
    GLMFamily          glm-5                                 (DSA + MoE)
    MoEFieldNormaliser any family: expert-count aliases       (field aliases)

The adapters are registered with the kernel's own registry, so the kernel
resolves them exactly as it resolves its built-ins — no parallel selection
logic, and `services/compatibility.py` keeps asking the kernel which
adapter it would use.

Accuracy rule: a family adapter must produce the *same* numbers as the
kernel's own adapter for a config the kernel already handles. The family
adapters are therefore written as thin extensions of the kernel's specs,
not as reimplementations of its formulas. `tests/test_family_adapters.py`
checks that equivalence.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "register_all",
    "FAMILY_MODULES",
]


def register_all() -> list[str]:
    """Register every family adapter with the kernel's registry.

    Returns the names of the adapters that were registered. Calling this
    more than once is safe: the kernel's registry rejects a duplicate
    registration of the same name, so the second call is a no-op.
    """
    from .qwen_hybrid import QwenHybridAdapter
    from .deepseek import DeepSeekFamilyAdapter
    from .glm import GLMFamilyAdapter
    from .moe_fields import MoEFieldNormalisingAdapter

    registered: list[str] = []
    for factory in (
        QwenHybridAdapter,
        DeepSeekFamilyAdapter,
        GLMFamilyAdapter,
        MoEFieldNormalisingAdapter,
    ):
        adapter = factory()
        registered.extend(_register(adapter))
    return registered


def _register(adapter: Any) -> list[str]:
    """Register one adapter, tolerating a duplicate registration."""
    from hcu_train_simulator.models.registry import register_model_adapter

    try:
        register_model_adapter(adapter)
    except ValueError:
        # Already registered — the kernel's registry raises on a duplicate
        # name, which is the signal that a previous call did the work.
        return []
    return [adapter.name]


#: The family modules, for display in the interface.
FAMILY_MODULES = [
    {
        "name": "QwenHybridFamily",
        "adapter": "qwen_hybrid",
        "covers": "Qwen3.5 / Qwen3-Next / Qwen3.5-Next 系列",
        "capability": "混合注意力（线性注意力层 + 全注意力层）、MTP、MoE",
        "detects": ["layer_types", "full_attention_interval", "linear_conv_kernel_dim"],
    },
    {
        "name": "DeepSeekFamily",
        "adapter": "deepseek_family",
        "covers": "DeepSeek V2 / V3 / R1 系列",
        "capability": "MLA 低秩 KV、MoE、MTP",
        "detects": ["kv_lora_rank", "qk_nope_head_dim", "n_routed_experts"],
    },
    {
        "name": "GLMFamily",
        "adapter": "glm_family",
        "covers": "GLM-5 系列",
        "capability": "DSA 索引注意力、MoE、MTP",
        "detects": ["index_head_dim", "index_n_heads", "index_topk", "indexer_types"],
    },
    {
        "name": "MoEFieldNormaliser",
        "adapter": "moe_field_normalised",
        "covers": "任意把专家数写在别名下的 MoE 系列",
        "capability": "专家数字段别名归一（不改变注意力结构）",
        "detects": ["num_experts", "n_routed_experts", "num_local_experts",
                    "moe_intermediate_size"],
    },
]
