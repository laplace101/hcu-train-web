"""Whether the estimator kernel can model a given model at all.

The kernel ships a fixed set of model adapters. A configuration that lands
on an adapter which cannot represent it — a vision tower on a text-only
adapter, a missing MLA field set, an unknown expert layout — would either be
refused deep inside the simulator or, worse, be silently approximated.

This module makes that judgement *before* the estimate runs, so the user gets
a specific reason and a suggestion instead of a bare assertion message.
"""

from __future__ import annotations

from typing import Any

#: Adapters that can represent a vision tower.
VISION_CAPABLE_ADAPTERS = {"qwen3_vl", "qwen3_5"}

#: Adapters that expect Multi-Latent Attention. They read these three fields
#: to decide whether the model uses MLA.
MLA_FIELDS = ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim")

#: Fields that only exist in architectures the kernel does not model. Their
#: presence means the model is structurally different from anything the
#: kernel has a profile for.
UNKNOWN_ARCH_MARKERS = {
    "dspark_n_routed_experts": "dspark 稀疏注意力结构",
    "dspark_num_experts_per_tok": "dspark 稀疏注意力结构",
    "engram_head_dim": "engram n-gram 记忆结构",
    "engram_n_heads": "engram n-gram 记忆结构",
    "candidate_block_size": "candidate 块稀疏结构",
    "index_head_dim": "DSA 索引注意力结构",
    "index_n_heads": "DSA 索引注意力结构",
    "index_topk": "DSA 索引注意力结构",
}


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    """The language part of a config, whether nested or flat."""
    text = config.get("text_config")
    if isinstance(text, dict):
        merged = dict(text)
        for key in ("architectures", "model_type", "tie_word_embeddings"):
            if key in config:
                merged.setdefault(key, config[key])
        return merged
    return config


def check_compatibility(config: dict[str, Any]) -> dict[str, Any]:
    """Decide whether the kernel can model this configuration.

    Returns a dict with `supported`, the reasons when it cannot, and a
    suggestion the UI can show.
    """
    reasons: list[str] = []
    suggestions: list[str] = []

    text = _text_config(config)
    # The top-level model_type describes the whole model; the nested one
    # only the language tower.
    model_type = str(config.get("model_type") or text.get("model_type") or "")
    architectures = [str(a) for a in (config.get("architectures") or [])]
    has_vision = isinstance(config.get("vision_config"), dict)

    # Which adapter would the kernel pick?
    adapter = _select_adapter(model_type, architectures)
    if adapter is None:
        reasons.append(
            f"内核没有匹配该模型类型的适配器（model_type={model_type!r}, "
            f"architectures={architectures}）"
        )
        suggestions.append(
            "可改用内核已支持的模型，或上传该模型的 config.json 并对照已支持模型的参数自行估算"
        )
        return {
            "supported": False,
            "adapter": None,
            "model_type": model_type,
            "has_vision": has_vision,
            "reasons": reasons,
            "suggestions": suggestions,
        }

    # A vision tower needs an adapter that can represent it.
    if has_vision and adapter not in VISION_CAPABLE_ADAPTERS:
        reasons.append(
            f"该模型是多模态模型（含 vision_config，{len(architectures) or '?'} 个架构声明），"
            f"而内核把它匹配到了文本适配器 {adapter!r}；"
            f"内核目前只有 {'、'.join(sorted(VISION_CAPABLE_ADAPTERS))} 支持视觉塔"
        )
        suggestions.append(
            "内核尚未提供 DeepSeek 系多模态适配器，无法给出可信估算；"
            "如需评估该模型，请等待内核支持，或先评估其纯文本版本"
        )

    # MLA-based adapters read these fields to pick the attention layout.
    if adapter == "deepseek_v3":
        if not text.get("kv_lora_rank") or not text.get("qk_nope_head_dim"):
            reasons.append(
                "内核的 DeepSeek 适配器依赖 MLA 字段（kv_lora_rank / qk_nope_head_dim / "
                "qk_rope_head_dim），而该模型缺少这些字段，注意力布局无法确定"
            )
            suggestions.append("内核未建模该模型采用的注意力变体")

    # Structural features the kernel has no profile for.
    markers = sorted({
        label for field, label in UNKNOWN_ARCH_MARKERS.items() if field in text
    })
    if markers:
        reasons.append(
            f"该模型包含内核未建模的结构：{'、'.join(markers)}；"
            "这些结构会改变算子组成与显存分布，按已知适配器计算会失真"
        )
        suggestions.append("内核尚未建模该架构变体，不提供估算以免给出失真数字")

    # The MoE layout fields the DeepSeek adapter validates.
    if adapter == "deepseek_v3":
        if "moe_layer_freq" not in text and "first_k_dense_replace" not in text:
            if text.get("n_routed_experts"):
                reasons.append(
                    "缺少 MoE 层分布参数（moe_layer_freq / first_k_dense_replace），"
                    "无法确定稠密层与专家层的划分"
                )

    supported = not reasons
    return {
        "supported": supported,
        "adapter": adapter,
        "model_type": model_type,
        "has_vision": has_vision,
        "reasons": reasons,
        "suggestions": suggestions,
        "unknown_features": markers,
    }


def _select_adapter(model_type: str, architectures: list[str]) -> str | None:
    """Mirror the kernel's adapter selection, for the compatibility check.

    The kernel scores an exact `model_type` match above a prefix match,
    and an architecture prefix above both. Reproducing it here keeps the
    message consistent with what the simulator would actually do.
    """
    exact = {
        "deepseek_v3": ("deepseek_v3", "deepseek_v2"),
        "qwen3": ("qwen3", "qwen3_moe"),
        "glm5": ("glm_moe_dsa",),
    }
    for adapter, types in exact.items():
        if model_type in types:
            return adapter

    prefixes = [
        ("qwen3_5", "qwen3_5"),
        ("qwen3_vl", "qwen3_vl"),
        ("qwen3", "qwen3"),
        ("deepseek_v3", "deepseek"),
        ("glm5", "glm"),
    ]
    for adapter, prefix in prefixes:
        if model_type.startswith(prefix):
            return adapter
        if any(name.startswith(prefix) for name in architectures):
            return adapter
    return None


def describe(config: dict[str, Any]) -> str:
    """One-line summary of the configuration's shape, for the UI."""
    text = _text_config(config)
    parts = [str(text.get("model_type") or "?")]
    if text.get("num_hidden_layers"):
        parts.append(f"{text['num_hidden_layers']} 层")
    if text.get("hidden_size"):
        parts.append(f"hidden {text['hidden_size']}")
    if text.get("n_routed_experts"):
        parts.append(f"{text['n_routed_experts']} 专家 MoE")
    if isinstance(config.get("vision_config"), dict):
        parts.append("多模态")
    return " · ".join(parts)
