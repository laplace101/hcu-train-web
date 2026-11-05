"""Model support matrix (plan §6 — model catalogue and compatibility).

The kernel models a fixed set of architectures. This service answers, for a
given model reference, whether the kernel can estimate it and — when it
cannot — exactly what is missing.

The answer is not a hard-coded list. It comes from two sources:

  * `services.compatibility` — which adapter the kernel would pick, and
    whether that adapter can represent the model's structures. The adapter
    is resolved by the kernel itself, so the answer agrees with what the
    simulator would actually do.
  * `catalog/catalog.json` — the preset models already in the repository,
    with the architecture each one is known to work with.

The distinction the matrix draws:

    estimable      — the kernel models this architecture; numbers are
                     computed with that architecture's formulas.
    approximate    — the kernel falls back to the generic adapter. Correct
                     for a plain dense decoder; an approximation for
                     anything with a different attention or expert layout.
    not_estimable  — the kernel cannot represent the model. The reasons
                     name what is missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services import compatibility

#: Where the preset catalogue lives.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "catalog" / "catalog.json"

#: The kernel's adapters, and what each one means.
ADAPTER_DESCRIPTIONS = {
    "qwen3": "Qwen3 稠密与 MoE，含 GQA 分组注意力",
    "qwen3_vl": "Qwen3-VL 多模态，视觉塔 + 文本解码器",
    "qwen3_5": "Qwen3.5 混合线性注意力，含逐层类型与 MTP",
    "deepseek_v3": "DeepSeek V2/V3，MLA 低秩 KV + MoE",
    "glm5": "GLM-5，DSA 索引注意力 + MoE",
    "generic_gpt": "通用稠密解码器兜底（标准注意力 + 标准 FFN）",
    # The family adapters this project adds.
    "qwen_hybrid": "Qwen 混合注意力系列（Qwen3.5 / Qwen3-Next），线性注意力与全注意力逐层交替",
    "deepseek_family": "DeepSeek 系列（V2/V3/R1），MLA + MoE，DSA 稀疏注意力",
    "glm_family": "GLM 系列（GLM-5），DSA 索引注意力 + MoE",
    "moe_field_normalised": "MoE 字段别名归一：把专家数写在别名下的系列",
    "minimax_family": "MiniMax 系列：sigmoid 门控 MoE、专家 FFN 宽度、MTP 模块、逐层注意力类型",
}


@dataclass
class ModelSupport:
    """The verdict for one model."""

    reference: str
    status: str                       # estimable | approximate | not_estimable
    adapter: str | None = None
    model_type: str | None = None
    architectures: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unknown_features: list[str] = field(default_factory=list)
    is_approximate: bool = False
    source: str | None = None
    revision: str | None = None
    preset: bool = False
    family: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "status": self.status,
            "status_label": STATUS_LABELS.get(self.status, self.status),
            "adapter": self.adapter,
            "adapter_description": ADAPTER_DESCRIPTIONS.get(self.adapter or "", ""),
            "model_type": self.model_type,
            "architectures": self.architectures,
            "reasons": self.reasons,
            "suggestions": self.suggestions,
            "warnings": self.warnings,
            "unknown_features": self.unknown_features,
            "is_approximate": self.is_approximate,
            "source": self.source,
            "revision": self.revision,
            "preset": self.preset,
            "family": self.family,
        }


STATUS_LABELS = {
    "estimable": "可精确估算",
    "approximate": "可估算（近似值）",
    "not_estimable": "不可估算",
}


def _classify(verdict: dict[str, Any]) -> str:
    """Map a compatibility verdict onto a support status.

    Three levels, and the middle one matters:

      * `estimable` — the kernel models this architecture exactly;
      * `approximate` — the kernel has no dedicated model for it, or the
        family adapter handles part of it only approximately (MiniMax's
        Lightning attention is mapped onto the kernel's Gated DeltaNet
        spec, which is a different operator);
      * `not_estimable` — the kernel cannot represent the model at all.
    """
    if not verdict["supported"]:
        return "not_estimable"
    if verdict.get("is_approximate"):
        return "approximate"
    if verdict.get("adapter_is_fallback"):
        return "approximate"
    return "estimable"


def _catalog_index() -> tuple[dict[str, dict], dict[str, dict]]:
    """Load the preset catalogue, indexed by slug and by name."""
    if not CATALOG_PATH.exists():
        return {}, {}
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    by_slug: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    for entry in data.get("models", []):
        by_slug[entry["slug"]] = entry
        name = entry.get("name") or entry["slug"]
        by_name[name.lower()] = entry
        hf_id = entry.get("hf_id")
        if hf_id:
            by_name[hf_id.lower()] = entry
    return by_slug, by_name


def check_reference(
    reference: str,
    config: dict[str, Any] | None = None,
    source: str | None = None,
    revision: str | None = None,
) -> ModelSupport:
    """Determine whether a model can be estimated.

    `config` may be supplied directly; otherwise the preset catalogue is
    consulted and the model is looked up by slug or by name.
    """
    reference = (reference or "").strip()
    support = ModelSupport(reference=reference, status="not_estimable", source=source,
                           revision=revision)

    if config is None:
        by_slug, by_name = _catalog_index()
        entry = by_slug.get(reference) or by_name.get(reference.lower())
        if entry is None:
            support.reasons = [
                "未找到该模型的配置；请提供 config.json，"
                "或先从模型社区拉取后再估算"
            ]
            support.suggestions = [
                "在「模型来源」中填入模型社区仓库名，或直接上传 config.json"
            ]
            return support
        config_path = PROJECT_ROOT / "catalog" / "models" / entry["slug"] / "config.json"
        if not config_path.exists():
            support.reasons = [f"预设模型 {entry['slug']} 的 config.json 不存在"]
            return support
        config = json.loads(config_path.read_text(encoding="utf-8"))
        support.preset = True
        support.family = entry.get("family")
        source = source or "catalog"
        support.source = source

    verdict = compatibility.check_compatibility(config)
    support.status = _classify(verdict)
    support.adapter = verdict.get("adapter")
    support.model_type = verdict.get("model_type")
    support.architectures = verdict.get("architectures") or []
    support.reasons = verdict.get("reasons") or []
    support.suggestions = verdict.get("suggestions") or []
    support.warnings = verdict.get("warnings") or []
    support.unknown_features = verdict.get("unknown_features") or []
    support.is_approximate = bool(verdict.get("is_approximate"))
    return support


#: How the upstream simulator treats a model it has no adapter for.
#:
#: `hcu-train-simulator` ships a `generic_gpt` fallback adapter with
#: `fallback = True`, and its README states that an unknown model family
#: "degrades to a generic Transformer/MoE scale approximation" with
#: unrecognised fields ignored. So upstream does produce numbers for
#: models it does not model — it just does not say how wrong they are.
#:
#: The compatibility gate in `services.compatibility` refuses some of those
#: models instead. This table records which behaviour applies, so the
#: interface can show the difference rather than hide it.
UPSTREAM_BASELINE = {
    "adapter_count": 6,
    "adapters": ["qwen3", "qwen3_vl", "qwen3_5", "deepseek_v3", "glm5",
                 "generic_gpt"],
    "fallback_adapter": "generic_gpt",
    "fallback_behaviour": (
        "未知模型族降级为通用 Transformer/MoE 规模近似，"
        "无法识别的额外字段被忽略并在终端打印假设"
    ),
    "gate_behaviour": (
        "兼容性门禁在估算之前拒绝内核无法表示的模型，"
        "并列出具体缺失项，不返回失真数字"
    ),
    "gate_note": (
        "两者的差别不在模型数量，而在失真模型的处理方式。"
        "内核的 generic 适配器对标准稠密解码器是正确的，"
        "对注意力或专家布局不同的模型则不是——"
        "例如 256 个专家的 MoE 被按一个稠密 FFN 计算"
    ),
}


def _family_modules() -> list[dict[str, Any]]:
    """The family modules this project adds on top of the kernel."""
    try:
        from engine import families

        return [dict(module) for module in families.FAMILY_MODULES]
    except Exception:  # noqa: BLE001
        return []


def upstream_baseline() -> dict[str, Any]:
    """The upstream simulator's own model coverage, for comparison."""
    return dict(UPSTREAM_BASELINE)


def matrix() -> dict[str, Any]:
    """The full support matrix over the preset catalogue.

    Every preset model is checked against the compatibility rules, so the
    matrix reflects what the kernel can actually do rather than a stored
    opinion.
    """
    by_slug, _ = _catalog_index()
    rows: list[dict[str, Any]] = []
    for slug, entry in sorted(by_slug.items()):
        support = check_reference(slug)
        support.family = entry.get("family")
        support.preset = True
        rows.append(support.to_dict())

    families: dict[str, dict[str, int]] = {}
    for row in rows:
        family = row.get("family") or "其他"
        bucket = families.setdefault(family, {"estimable": 0, "approximate": 0,
                                              "not_estimable": 0})
        bucket[row["status"]] += 1

    return {
        "adapters": [
            {"name": name, "description": description}
            for name, description in ADAPTER_DESCRIPTIONS.items()
        ],
        "upstream": upstream_baseline(),
        "family_modules": _family_modules(),
        "families": families,
        "models": rows,
        "totals": {
            "estimable": sum(1 for r in rows if r["status"] == "estimable"),
            "approximate": sum(1 for r in rows if r["status"] == "approximate"),
            "not_estimable": sum(1 for r in rows if r["status"] == "not_estimable"),
            "total": len(rows),
        },
    }


#: The config fields each adapter reads. Taken from the kernel's adapter
#: sources, so the matrix states what the kernel actually consumes.
ADAPTER_FIELDS: dict[str, set[str]] = {
    "minimax_family": {"num_local_experts", "num_experts",
                       "mlp_intermediate_size", "moe_intermediate_size",
                       "num_experts_per_tok", "num_mtp_modules",
                       "mtp_num_layers", "attn_type_list", "layer_types",
                       "scoring_func", "shared_moe_mode", "use_routing_bias",
                       "use_mtp", "mtp_transformer_layers"},
    "qwen_hybrid": {"layer_types", "full_attention_interval",
                    "linear_conv_kernel_dim", "linear_key_head_dim",
                    "linear_value_head_dim", "linear_num_key_heads",
                    "linear_num_value_heads", "partial_rotary_factor",
                    "num_experts", "num_experts_per_tok"},
    "deepseek_family": {"kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
                        "q_lora_rank", "index_head_dim", "index_n_heads",
                        "index_topk", "indexer_types", "moe_layer_freq",
                        "first_k_dense_replace", "partial_rotary_factor",
                        "n_routed_experts", "num_experts"},
    "glm_family": {"index_head_dim", "index_n_heads", "index_topk",
                   "indexer_types", "moe_layer_freq", "first_k_dense_replace",
                   "n_routed_experts", "num_experts"},
    "moe_field_normalised": {"num_experts", "n_routed_experts",
                             "num_local_experts", "moe_intermediate_size",
                             "num_experts_per_tok"},
    "qwen3": {"num_experts", "num_experts_per_tok", "num_shared_experts"},
    "qwen3_vl": {"hidden_size", "num_experts", "num_experts_per_tok",
                 "moe_layer_freq", "moe_dense_layer_indices"},
    "qwen3_5": {"layer_types", "full_attention_interval", "mtp_num_layers",
                "num_experts", "num_experts_per_tok"},
    "deepseek_v3": {"kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
                    "moe_layer_freq", "first_k_dense_replace", "mtp_num_layers",
                    "n_routed_experts"},
    "glm5": {"index_head_dim", "index_n_heads", "index_topk", "indexer_types",
             "q_lora_rank", "qk_rope_head_dim", "moe_layer_freq",
             "first_k_dense_replace", "n_routed_experts"},
    "generic_gpt": {"kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
                    "num_experts", "n_routed_experts"},
}


def adapter_capabilities() -> dict[str, Any]:
    """What each adapter needs from a model config.

    Used by the interface to explain why a model was rejected.
    """
    return {
        name: {
            "description": description,
            "reads": sorted(ADAPTER_FIELDS.get(name, set())),
            "supports_vision": name in {"qwen3_vl", "qwen3_5", "qwen_hybrid"},
            "supports_dsa": name in {"glm5", "deepseek_family", "glm_family"},
            "supports_mlp_attention": name in {"qwen3_5", "qwen_hybrid"},
            "supports_hybrid_attention": name in {"qwen3_5", "qwen_hybrid",
                                                  "minimax_family"},
            "is_fallback": name in compatibility.FALLBACK_ADAPTERS,
            "is_family_module": name in compatibility.FAMILY_ADAPTERS,
        }
        for name, description in ADAPTER_DESCRIPTIONS.items()
    }
