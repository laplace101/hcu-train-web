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

#: The kernel's fallback adapter. It models a plain transformer decoder, so a
#: model that lands here is estimated with the generic formulas — correct for
#: an ordinary dense decoder, but not for an architecture whose attention or
#: expert layout differs from the standard one.
FALLBACK_ADAPTERS = {"generic_gpt"}

#: Adapters with a family-specific model spec. A model handled by one of
#: these is estimated with that family's layout (MLA, DSA, fused kernels …).
SPECIALISED_ADAPTERS = {"qwen3", "qwen3_vl", "qwen3_5", "deepseek_v3", "glm5"}

#: Adapters that expect Multi-Latent Attention. They read these three fields
#: to decide whether the model uses MLA.
MLA_FIELDS = ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim")

#: Config features the kernel does not model but that only affect the
#: constants, not the attention layout. Worth reporting, not worth refusing:
#: the estimate stays the right order of magnitude.
UNMODELLED_BUT_HARMLESS = {
    "sliding_window": "滑窗注意力（内核按全注意力计算）",
    "rope_scaling": "RoPE 缩放（内核按默认 RoPE 计算）",
}

#: Structures the kernel has no config field for at all, keyed by the
#: field that signals them. These are the real blockers: the model is
#: architecturally different from anything the kernel describes.
#:
#: `index_*` is the DSA (DeepSeek Sparse Attention) layout. The kernel
#: reads those fields, but only the `glm5` adapter applies them, so a DSA
#: model is fine when it lands there.
UNKNOWN_ARCH_MARKERS = {
    "dspark_n_routed_experts": "dspark 稀疏注意力结构",
    "dspark_num_experts_per_tok": "dspark 稀疏注意力结构",
    "engram_head_dim": "engram n-gram 记忆结构",
    "engram_n_heads": "engram n-gram 记忆结构",
    "candidate_block_size": "candidate 块稀疏结构",
    "attn_type_list": "混合线性/全注意力（Lightning Attention）",
    "use_mtp": "多 token 预测（MTP）开关",
    "mtp_transformer_layers": "多 token 预测（MTP）层定义",
    "index_head_dim": "DSA 索引注意力结构",
    "index_n_heads": "DSA 索引注意力结构",
    "index_topk": "DSA 索引注意力结构",
}

#: Adapters whose model spec includes the DSA index-attention layout.
DSA_CAPABLE_ADAPTERS = {"glm5"}

#: The config fields the kernel reads an expert count from. It takes the
#: first of `num_experts` and `n_routed_experts`; a model that spells its
#: expert count some other way (`num_local_experts`, used by MiniMax) is
#: modelled as a dense decoder, which understates its memory by a wide
#: margin.
EXPERT_COUNT_FIELDS = ("num_experts", "n_routed_experts")

#: An expert count written under a name the kernel does not read.
UNREADABLE_EXPERT_FIELDS = ("num_local_experts",)

#: Config fields each adapter reads. A field that only some adapter
#: understands is modelled there and unmodelled elsewhere — `layer_types`,
#: for instance, is read by the `qwen3_5` adapter and by no other, so it is
#: not a reason to refuse a Qwen3.5 model.
ADAPTER_READS_FIELDS = {
    "layer_types": {"qwen3_5"},
    "full_attention_interval": {"qwen3_5"},
    "linear_conv_kernel_dim": {"qwen3_5"},
    "linear_key_head_dim": {"qwen3_5"},
    "linear_value_head_dim": {"qwen3_5"},
    "linear_num_key_heads": {"qwen3_5"},
    "linear_num_value_heads": {"qwen3_5"},
    "attn_output_gate": {"qwen3_5"},
    "attention_output_gate": {"qwen3_5"},
    "partial_rotary_factor": {"deepseek_v3", "glm5"},
    "index_head_dim": {"glm5"},
    "index_n_heads": {"glm5"},
    "index_topk": {"glm5"},
    "indexer_types": {"glm5"},
    "kv_lora_rank": {"deepseek_v3", "generic_gpt"},
    "qk_nope_head_dim": {"deepseek_v3", "generic_gpt"},
    "qk_rope_head_dim": {"deepseek_v3", "generic_gpt"},
    "q_lora_rank": {"glm5"},
    "first_k_dense_replace": {"deepseek_v3", "glm5", "qwen3_vl"},
    "moe_layer_freq": {"deepseek_v3", "glm5", "qwen3_vl"},
    "moe_dense_layer_indices": {"qwen3_vl"},
    "mtp_num_layers": {"qwen3_5", "deepseek_v3", "glm5"},
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
    warnings: list[str] = []

    text = _text_config(config)
    # The top-level model_type describes the whole model; the nested one
    # only the language tower.
    model_type = str(config.get("model_type") or text.get("model_type") or "")
    architectures = [str(a) for a in (config.get("architectures") or [])]
    has_vision = isinstance(config.get("vision_config"), dict)

    # Which adapter would the kernel pick? The kernel itself is the
    # authority — see _select_adapter.
    adapter = _select_adapter(config)
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
            "adapter_is_fallback": False,
            "adapter_is_specialised": False,
            "model_type": model_type,
            "architectures": architectures,
            "has_vision": has_vision,
            "reasons": reasons,
            "suggestions": suggestions,
            "warnings": warnings,
            "unknown_features": [],
            "unapplied_fields": [],
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


    # The MoE layout fields the DeepSeek adapter validates.
    if adapter == "deepseek_v3":
        if "moe_layer_freq" not in text and "first_k_dense_replace" not in text:
            if text.get("n_routed_experts"):
                reasons.append(
                    "缺少 MoE 层分布参数（moe_layer_freq / first_k_dense_replace），"
                    "无法确定稠密层与专家层的划分"
                )

    # An expert count the kernel cannot read makes the model look dense.
    if not any(text.get(f) for f in EXPERT_COUNT_FIELDS):
        expert_field = next(
            (f for f in UNREADABLE_EXPERT_FIELDS if text.get(f)), None
        )
        if expert_field:
            reasons.append(
                f"该模型的专家数写在 {expert_field}={text.get(expert_field)}，"
                f"而内核只读 {' 或 '.join(EXPERT_COUNT_FIELDS)}，"
                "会把它当成稠密模型计算，显存与算力会被大幅低估"
            )
            suggestions.append(
                "内核缺少该架构的 MoE 建模，不提供估算；"
                "如需评估，请等待内核支持该模型的适配器"
            )

    # Structures the kernel has no config field for. These block the
    # estimate: the model is architecturally different from anything the
    # kernel describes.
    unsupported = sorted({
        label for field, label in UNKNOWN_ARCH_MARKERS.items()
        if text.get(field) or config.get(field)
    })
    # The DSA layout is read by the kernel but only applied by the `glm5`
    # adapter, so a DSA model is fine when it lands there.
    if adapter in DSA_CAPABLE_ADAPTERS:
        unsupported = [m for m in unsupported if m != "DSA 索引注意力结构"]
    if unsupported:
        reasons.append(
            f"该模型包含内核无法表示的结构：{'、'.join(unsupported)}"
        )
        suggestions.append("内核尚未建模该架构特征，不提供估算")

    # Features the kernel does not model, but which do not change the
    # attention layout or the expert distribution. These are reported so
    # the estimate is not mistaken for an exact one.
    minor = sorted({
        label for field, label in UNMODELLED_BUT_HARMLESS.items()
        if text.get(field) is not None or config.get(field) is not None
    })
    if minor:
        warnings.append(
            f"该模型带有内核未建模的次要特性：{'、'.join(minor)}；"
            "不影响注意力布局与专家分布，估算结果仍在同一量级"
        )

    # Fields the kernel reads but the chosen adapter does not apply. These
    # do not block the estimate — the structure is approximated with the
    # adapter's own formulas — so they are reported as warnings.
    unapplied = sorted({
        field for field, readers in ADAPTER_READS_FIELDS.items()
        if text.get(field) is not None and adapter not in readers
    })
    if unapplied:
        warnings.append(
            f"适配器 {adapter} 不使用模型配置中的 "
            f"{'、'.join(unapplied)} 字段，相关结构按该适配器的通用公式处理"
        )

    # A model with family-specific structures that lands on the generic
    # fallback will be estimated with the standard decoder formulas. That is
    # an approximation, and it is reported as such.
    if adapter in FALLBACK_ADAPTERS and unsupported:
        warnings.append(
            f"该模型的结构特征（{'、'.join(unsupported)}）在内核里没有专用适配器，"
            f"只能落到通用兜底适配器 {adapter} 按标准稠密解码器计算，结果为近似值"
        )
        suggestions.append(
            "如需精确结果，需要内核为该架构提供专用适配器；"
            "当前结果可用于量级参考，不宜作为容量规划的最终依据"
        )

    supported = not reasons
    return {
        "supported": supported,
        "adapter": adapter,
        "adapter_is_fallback": adapter in FALLBACK_ADAPTERS,
        "adapter_is_specialised": adapter in SPECIALISED_ADAPTERS,
        "model_type": model_type,
        "architectures": architectures,
        "has_vision": has_vision,
        "reasons": reasons,
        "suggestions": suggestions,
        "warnings": warnings,
        # The structures the kernel has no representation for.
        "unknown_features": unsupported,
        # The config fields the chosen adapter does not apply.
        "unapplied_fields": unapplied,
    }


def _select_adapter(config: dict[str, Any]) -> str | None:
    """Ask the kernel which adapter it would use for this config.

    The kernel's own `resolve_model_adapter` is the authority — it scores
    exact model-type matches above prefix matches and architecture prefixes
    above both, and falls back to the generic adapter for anything else.
    Reimplementing that here was wrong twice: it missed the fallback and it
    matched `glm4_moe` to the `glm5` adapter.

    The simulator source is located the same way the rest of the project
    locates it; if it cannot be found, the check falls back to a local
    approximation and says so.
    """
    try:
        from engine import paths

        # `paths.bootstrap()` imports the simulator package in the order the
        # kernel requires. Importing `models.registry` directly triggers a
        # circular import between the qwen3_5 and qwen_vl benchmark modules,
        # which made this fall back to the local approximation and report
        # the wrong adapter.
        paths.bootstrap()
        from hcu_train_simulator.models.registry import resolve_model_adapter

        adapter = resolve_model_adapter(config)
        return adapter.name
    except Exception:  # noqa: BLE001
        return _select_adapter_local(config)


def _select_adapter_local(config: dict[str, Any]) -> str | None:
    """Approximate the kernel's adapter selection without the kernel.

    Only used when the simulator source is not available. The generic
    adapter is a fallback in the kernel, so an unknown model type is still
    handled — but by the generic adapter, not a family-specific one.
    """
    model_type = str(config.get("model_type") or "")
    architectures = [str(a) for a in (config.get("architectures") or [])]

    exact = {
        "deepseek_v3": ("deepseek_v3", "deepseek_v2"),
        "qwen3": ("qwen3", "qwen3_moe"),
        "qwen3_5": ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"),
        "glm5": ("glm_moe_dsa",),
    }
    for adapter, types in exact.items():
        if model_type in types:
            return adapter

    prefixes = [
        ("qwen3_5", "qwen3_5"),
        ("qwen3_vl", "qwen3_vl"),
        ("qwen3", "qwen3_"),
        ("deepseek_v3", "deepseek"),
        ("glm5", "glm_moe_dsa"),
    ]
    for adapter, prefix in prefixes:
        if model_type.startswith(prefix):
            return adapter
        if any(name.startswith(prefix) for name in architectures):
            return adapter

    # The kernel's generic adapter is a fallback.
    return "generic_gpt"


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
