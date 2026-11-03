"""Preset model catalog access and model-config inspection."""

from __future__ import annotations

import json
from typing import Any

from . import paths


def load_catalog() -> list[dict[str, Any]]:
    if not paths.CATALOG_FILE.exists():
        return []
    try:
        data = json.loads(paths.CATALOG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    models = data.get("models") if isinstance(data, dict) else data
    return models or []


def get_entry(slug: str) -> dict[str, Any] | None:
    for entry in load_catalog():
        if entry.get("slug") == slug:
            return entry
    return None


def family_summary(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order: list[str] = []
    counts: dict[str, int] = {}
    for model in models:
        family = model.get("family") or "其他"
        if family not in counts:
            order.append(family)
            counts[family] = 0
        counts[family] += 1
    return [{"name": name, "count": counts[name]} for name in order]


# ---------------------------------------------------------------------------
# Architecture summary for an arbitrary config.json
# ---------------------------------------------------------------------------

def _text_config(raw: dict[str, Any]) -> dict[str, Any]:
    nested = raw.get("text_config")
    if isinstance(nested, dict):
        merged = dict(nested)
        for key in ("architectures", "model_type", "tie_word_embeddings"):
            if key in raw:
                merged[key] = raw[key]
        return merged
    return dict(raw)


def _first(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


def inspect_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Describe a model config the same way the simulator will interpret it."""

    text = _text_config(raw)
    model_type = raw.get("model_type") or text.get("model_type")
    vision = raw.get("vision_config") if isinstance(raw.get("vision_config"), dict) else None

    experts = _first(text, "num_experts", "n_routed_experts")
    topk = _first(text, "num_experts_per_tok", "num_experts_per_token", "topk")
    shared = _first(text, "num_shared_experts", "n_shared_experts")
    q_lora = _first(text, "q_lora_rank")
    kv_lora = _first(text, "kv_lora_rank")
    mtp = _first(text, "mtp_num_layers", "num_mtp_layers", "num_nextn_predict_layers", default=0)

    adapter = None
    adapter_error = None
    try:
        paths.bootstrap()
        from hcu_train_simulator.models.registry import resolve_model_adapter

        adapter = resolve_model_adapter(raw).name
    except Exception as exc:  # noqa: BLE001
        adapter_error = str(exc)

    heads = _first(text, "num_attention_heads", default=0)
    kv_groups = _first(text, "num_key_value_heads", "num_kv_heads", default=heads)

    traits: list[str] = []
    if vision:
        traits.append("视觉编码器")
    if experts:
        traits.append(f"MoE（{experts} 专家 / top-{topk}）")
    if shared:
        traits.append(f"共享专家 ×{shared}")
    if q_lora and kv_lora:
        traits.append("MLA")
    if kv_groups and heads and kv_groups != heads:
        traits.append(f"GQA（{heads} 头 / {kv_groups} KV 组）")
    if mtp:
        traits.append(f"MTP ×{mtp}")
    if text.get("model_type", "").startswith("glm_moe_dsa") or raw.get("index_topk"):
        traits.append("DSA 稀疏注意力")
    if not traits:
        traits.append("标准稠密 Transformer")

    return {
        "model_type": model_type,
        "architectures": raw.get("architectures") or text.get("architectures"),
        "adapter": adapter,
        "adapter_error": adapter_error,
        "adapter_is_fallback": adapter == "generic_gpt",
        "num_hidden_layers": _first(text, "num_hidden_layers", default=0),
        "hidden_size": _first(text, "hidden_size", default=0),
        "num_attention_heads": heads,
        "num_query_groups": kv_groups,
        "vocab_size": _first(text, "vocab_size", default=0),
        "intermediate_size": _first(text, "intermediate_size", "ffn_hidden_size"),
        "num_experts": experts,
        "num_experts_per_tok": topk,
        "mtp_num_layers": mtp,
        "has_vision": bool(vision),
        "vision_layers": (vision or {}).get("depth", (vision or {}).get("num_hidden_layers", 0)),
        "is_moe": bool(experts),
        "traits": traits,
        "tp_divisors": _tp_divisors(_first(text, "hidden_size", default=0), heads, kv_groups),
        "pp_divisors": _pp_divisors(_first(text, "num_hidden_layers", default=0)),
    }


def _tp_divisors(hidden: int, heads: int, kv_groups: int) -> list[int]:
    options = []
    for tp in (1, 2, 4, 8):
        if heads and heads % tp == 0 and kv_groups and kv_groups % tp == 0:
            if not hidden or hidden % tp == 0:
                options.append(tp)
    return options or [1]


def _pp_divisors(layers: int) -> list[int]:
    if not layers:
        return [1]
    return [pp for pp in (1, 2, 4, 8) if layers % pp == 0] or [1]