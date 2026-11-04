"""Training estimate service (plan §3, D01/D04).

Turns a `TrainingScenario` into simulator configurations and returns the three
plan types the product defines:

  1. 最低可训练资源方案  — minimum card count that fits, with a safety margin
  2. 推荐稳定方案       — keeps more headroom, avoids pathological layouts
  3. 推荐性能方案       — best TGS within the given resource ceiling

The simulator's `solve` returns the first and a "higher performance" variant.
The stable plan is derived from the performance plan by checking that its
memory headroom and communication ratio are sane; when the performance plan is
too aggressive, the stable plan falls back to the minimum plan and says why.

Everything numeric comes from `estimators.adapter`; this module only shapes
the request and organises the answer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from dataclasses import replace

from domain.scenarios import MODE_SPECS, TrainingMode, TrainingScenario
from estimators import adapter
from estimators.adapter import UnsupportedModeError
from engine import config_builder
from repositories import measured as measured_repo
from repositories.hardware import HARDWARE_PROFILES, resolve_hardware
from services import lora_calibration, lora_model
from services.validation import has_errors, validate_scenario

APP_ROOT = Path(__file__).resolve().parents[1]

__all__ = ["estimate", "build_simulator_config", "model_facts", "load_model_facts",
           "ScenarioRejected", "UnsupportedModeError"]

#: Memory headroom (GiB) a plan must keep free on every card.
MIN_HEADROOM_GIB = 5.0

#: Communication share above which a plan is considered communication-bound
#: and therefore not "stable".
MAX_STABLE_COMM_RATIO = 0.5


class ScenarioRejected(Exception):
    """The scenario cannot produce an estimate; carries the reasons."""

    def __init__(self, message: str, issues: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.issues = issues or []


def _catalog_config_path(model_slug: str | None) -> str | None:
    """Locate a preset model's config.json in the catalogue."""
    if not model_slug:
        return None
    candidate = APP_ROOT / "catalog" / "models" / model_slug / "config.json"
    return str(candidate) if candidate.exists() else None


def load_model_facts(model_slug: str | None, config_path: str | None) -> dict[str, Any] | None:
    """Read the model facts needed for validation and display."""
    if config_path:
        path = Path(config_path)
        if path.is_dir():
            path = path / "config.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    if not model_slug:
        return None
    candidate = APP_ROOT / "catalog" / "models" / model_slug / "config.json"
    if candidate.exists():
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _flatten(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested (text_config / vision_config) HF config."""
    text = config.get("text_config")
    if not isinstance(text, dict):
        return dict(config)
    merged = dict(text)
    for key in ("architectures", "model_type", "tie_word_embeddings"):
        if key in config:
            merged[key] = config[key]
    return merged


def model_facts(config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the shape facts the validator and the UI need."""
    if not config:
        return {}
    text = _flatten(config)

    def first(*keys, default=None):
        for key in keys:
            value = text.get(key)
            if value is not None:
                return value
        return default

    return {
        "model_type": config.get("model_type") or text.get("model_type"),
        "architectures": config.get("architectures") or [],
        "num_hidden_layers": first("num_hidden_layers", default=0),
        "hidden_size": first("hidden_size", default=0),
        "intermediate_size": first("intermediate_size"),
        "moe_intermediate_size": first("moe_intermediate_size"),
        "num_attention_heads": first("num_attention_heads", default=0),
        "num_key_value_heads": first("num_key_value_heads", "num_query_groups", default=0),
        "head_dim": first("head_dim", default=0),
        "vocab_size": first("vocab_size", default=0),
        "n_routed_experts": first("n_routed_experts", "num_experts"),
        "num_experts_per_tok": first("num_experts_per_tok", "topk"),
        "n_shared_experts": first("n_shared_experts", "num_shared_experts"),
        "kv_lora_rank": first("kv_lora_rank"),
        "index_head_dim": first("index_head_dim"),
        "max_position_embeddings": first("max_position_embeddings", default=0),
        "is_vlm": isinstance(config.get("vision_config"), dict),
        "is_moe": bool(first("n_routed_experts", "num_experts")),
    }


def build_simulator_config(
    scenario: TrainingScenario,
    hardware_id: str,
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
    etp_size: int | None = None,
    num_gpus: int | None = None,
    micro_batch_size: int | None = None,
    global_batch_size: int | None = None,
    seq_length: int | None = None,
) -> dict[str, Any]:
    """Assemble the simulator config for a scenario.

    The config shape comes from `engine.config_builder.build_config` — the
    same builder the verified 35/35 accuracy baseline uses. Building it by
    hand here would risk drifting from those numbers, so this function only
    translates a scenario into the builder's inputs.

    `search` is enabled for the solve path so the simulator explores the
    candidate space itself; that keeps the plan search in one place instead
    of re-implementing it.
    """
    profile = resolve_hardware(hardware_id)
    if profile is None:
        raise ScenarioRejected(f"未知硬件型号：{hardware_id}")

    gpus = num_gpus or scenario.num_gpus
    gpn = min(profile.gpus_per_node, gpus)

    parallel: dict[str, Any] = {
        "tp_size": tp_size,
        "pp_size": pp_size,
        "cp_size": cp_size,
        "ep_size": ep_size,
        "num_gpus": gpus,
        "micro_batch_size": micro_batch_size or scenario.parallel.micro_batch_size,
        "global_batch_size": global_batch_size or scenario.parallel.global_batch_size,
        "seq_length": seq_length or scenario.parallel.seq_length,
        "use_distributed_optimizer": scenario.use_distributed_optimizer,
        "full_recompute": scenario.full_recompute,
        "use_fp8_training": scenario.use_fp8,
    }
    if etp_size is not None:
        parallel["etp_size"] = etp_size

    # `full` and `pretrain` are the two modelled modes; the task key selects
    # the task defaults (pretrain turns on full recompute).
    task_key = "pretrain" if scenario.mode is TrainingMode.PRETRAIN else "full"

    search_block = {
        "enabled": True,
        "top_k": 10,
        "memory_margin_gib": MIN_HEADROOM_GIB,
        "candidates": {
            "tp_size": [1, 2, 4, 8],
            "cp_size": [1],
            "pp_size": [1, 2, 4, 8],
            "ep_size": [1, 2, 4, 8],
            "etp_size": [1, 2, 4],
            "micro_batch_size": [1, 2, 4],
        },
    }

    return config_builder.build_config(
        scenario.model_config_path,
        task=task_key,
        hardware=hardware_id,
        operator_profile=profile.id,
        parallel=parallel,
        search=search_block,
    )


def _communication_ratio(plan: dict[str, Any]) -> float | None:
    iteration = plan.get("iteration_ms")
    comm = plan.get("communication_time_ms")
    if not iteration or comm is None:
        return None
    return comm / iteration


def _classify_plan(
    plan: dict[str, Any], hardware_gib: int
) -> tuple[str, list[str]]:
    """Label a plan as stable or not, with reasons.

    A plan is *stable* when every card keeps at least MIN_HEADROOM_GIB free
    and communication does not dominate the iteration.
    """
    reasons: list[str] = []
    peak = plan.get("peak_gib")
    if peak is not None and peak > hardware_gib - MIN_HEADROOM_GIB:
        reasons.append(
            f"峰值显存 {peak} GiB 逼近单卡容量 {hardware_gib} GiB，"
            f"安全余量不足 {MIN_HEADROOM_GIB} GiB"
        )

    ratio = _communication_ratio(plan)
    if ratio is not None and ratio > MAX_STABLE_COMM_RATIO:
        reasons.append(
            f"通信占比 {ratio * 100:.1f}% 偏高，迭代时间由通信主导"
        )

    return ("stable" if not reasons else "risky"), reasons


def _measured_match_key(
    row: dict[str, Any], gpus: int | None, seq: int | None, mbs: int
) -> tuple:
    """Sort key that prefers the row closest to the estimate.

    Priority: same card count, then the same sequence length, then the
    micro-batch size closest to the estimate, then the highest throughput.
    """
    return (
        0 if row.get("gpu_count") == gpus else 1,
        0 if row.get("cutoff_len") == seq else 1,
        abs((row.get("bs") or 0) - mbs),
        -(row.get("tokens_per_s") or 0),
    )


def measured_comparison(
    scenario: TrainingScenario,
    facts: dict[str, Any],
    plans: dict[str, Any],
    hardware: Any,
) -> dict[str, Any]:
    """Attach measured benchmark rows to an estimate.

    The measured data is LoRA fine-tuning on real BW1000 hardware, while the
    estimator models full-parameter training. The two are therefore *not*
    directly comparable, and the payload says so explicitly:

      * the throughput numbers are close enough to be a sanity check;
      * the memory numbers are not comparable at all, because LoRA freezes
        the base weights — no base gradients, no base optimiser state.

    Nothing here is presented as a validation of the estimate; it is
    reference data from the same hardware.
    """
    if not measured_repo.available():
        return {
            "available": False,
            "reason": "本机未找到实测数据库 bench.db",
        }

    # Which catalogue slug does this estimate correspond to?
    slug = scenario.model_slug
    if not slug:
        # Fall back to matching the catalogue by the model's architecture.
        slug = _match_slug_by_facts(facts)
    if not slug:
        return {
            "available": True,
            "matched": False,
            "reason": "该模型不在实测数据库的覆盖范围内",
        }

    summary = measured_repo.summarise_for_slug(slug)
    if not summary.get("rows"):
        return {
            "available": True,
            "matched": False,
            "slug": slug,
            "reason": summary.get("reason") or "实测库中没有该模型的成功记录",
        }

    rows = summary["rows"]
    gpus = scenario.num_gpus
    seq = scenario.parallel.seq_length

    same_gpus = [r for r in rows if r.get("gpu_count") == gpus]
    same_seq = [r for r in (same_gpus or rows) if r.get("cutoff_len") == seq]
    candidates = same_seq or same_gpus or rows

    mbs = scenario.parallel.micro_batch_size
    best = sorted(candidates, key=lambda r: _measured_match_key(r, gpus, seq, mbs))[0]

    # Compare the estimate against the measurement.
    estimate_tgs = None
    current = plans.get("current") or {}
    if current.get("tgs"):
        estimate_tgs = current["tgs"]
    elif plans.get("performance", {}).get("tgs"):
        estimate_tgs = plans["performance"]["tgs"]

    measured_tgs = best.get("tokens_per_s")
    deviation = None
    if estimate_tgs and measured_tgs:
        deviation = round((estimate_tgs - measured_tgs) / measured_tgs * 100, 1)

    return {
        "available": True,
        "matched": True,
        "slug": slug,
        "model_names": summary.get("model_names"),
        "source": summary.get("source"),
        "measured_is_lora": True,
        "comparability_note": (
            "实测数据为 LoRA 微调（基座冻结，无基座梯度与优化器状态），"
            "估算为全参训练内核。吞吐量可作量级参照，显存不可直接对比——"
            "LoRA 冻结基座后显存显著低于全参估算属预期现象。"
        ),
        "estimate_tgs": estimate_tgs,
        "measured_tgs": measured_tgs,
        "tgs_deviation_percent": deviation,
        "estimate_peak_gib": current.get("peak_gib"),
        "measured_peak_gib": (
            round(best["peak_mem_mib"] / 1024, 2) if best.get("peak_mem_mib") else None
        ),
        "best_match": best,
        "match_quality": _match_quality(best, gpus, seq, mbs),
        "match_bs": best.get("bs"),
        "estimate_bs": mbs,
        "row_count": summary.get("count"),
        "oom_count": summary.get("oom_count"),
        "gpu_counts": summary.get("gpu_counts"),
        "rows": rows[:20],
    }


def _match_quality(
    row: dict[str, Any], gpus: int | None, seq: int | None, mbs: int
) -> str:
    """How close the matched row is to the estimate's configuration.

    The micro-batch size drives throughput directly, so a row that only
    matches on cards and sequence length is reported separately instead of
    being presented as an equal-configuration comparison.
    """
    same_gpus = row.get("gpu_count") == gpus
    same_seq = row.get("cutoff_len") == seq
    same_bs = row.get("bs") == mbs
    if same_gpus and same_seq and same_bs:
        return "exact"
    if same_gpus and same_seq:
        return "cards_and_seq"
    if same_gpus:
        return "cards_only"
    return "model_only"


def _match_slug_by_facts(facts: dict[str, Any]) -> str | None:
    """Find a catalogue slug from the model's architecture facts."""
    if not facts:
        return None
    hidden = facts.get("hidden_size") or 0
    layers = facts.get("num_hidden_layers") or 0
    if not hidden:
        return None
    cat_dir = APP_ROOT / "catalog" / "models"
    if not cat_dir.exists():
        return None
    for path in sorted(cat_dir.iterdir()):
        config = path / "config.json"
        if not config.exists():
            continue
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        text = data.get("text_config") if isinstance(data.get("text_config"), dict) else data
        if (text.get("hidden_size") == hidden
                and text.get("num_hidden_layers") == layers):
            return path.name
    return None


def _catalog_slug_by_name(name: str | None) -> str | None:
    """Resolve a catalogue display name to its directory slug.

    `catalog.json` is the authority: it maps slug -> name -> hf_id. The
    normalised comparison covers "Qwen3.5-2B" vs "qwen3-5-2b".
    """
    if not name:
        return None
    raw = str(name)
    if "/" in raw and raw.endswith(".json"):
        parts = raw.replace("\\", "/").split("/")
        if "models" in parts:
            index = parts.index("models")
            if index + 1 < len(parts):
                return parts[index + 1]
    target = raw.strip().lower()
    flat = target.replace("-", "").replace(".", "").replace("_", "").replace("/", "")

    catalog_file = APP_ROOT / "catalog" / "catalog.json"
    if catalog_file.exists():
        try:
            data = json.loads(catalog_file.read_text(encoding="utf-8"))
            for entry in data.get("models", []):
                for field in ("slug", "name", "hf_id"):
                    value = entry.get(field)
                    if not value:
                        continue
                    if value.lower() == target:
                        return entry.get("slug")
                    if value.lower().replace("-", "").replace(".", "").replace("_", "").replace("/", "") == flat:
                        return entry.get("slug")
        except (OSError, json.JSONDecodeError):
            pass

    # Fall back to the directory names.
    cat_dir = APP_ROOT / "catalog" / "models"
    if cat_dir.exists():
        for path in sorted(cat_dir.iterdir()):
            if path.is_dir() and path.name.lower().replace("-", "").replace(".", "").replace("_", "") == flat:
                return path.name
    return None


def measured_for_legacy(
    config: dict[str, Any], source_info: dict[str, Any], outcome: dict[str, Any]
) -> dict[str, Any]:
    """Attach measured benchmark data to a legacy `/api/simulate` response.

    The legacy route has no `TrainingScenario`, so the pieces needed for a
    comparison are read straight out of the simulator config. The result is
    the same shape as `measured_comparison` returns, so the UI renders both
    identically.
    """
    if not measured_repo.available():
        return {"available": False, "reason": "本机未找到实测数据库 bench.db"}

    parallel = config.get("parallel_config") or {}
    gpus = parallel.get("num_gpus")
    seq = parallel.get("seq_length")
    mbs = parallel.get("micro_batch_size")

    # A preset model gives its slug directly; a custom path is matched by
    # architecture instead.
    slug = None
    if (source_info or {}).get("kind") == "preset":
        # The preset label is the display name; resolve it to the catalogue
        # directory slug so the measured lookup uses the same key.
        slug = _catalog_slug_by_name((source_info or {}).get("label"))

    facts = model_facts(load_model_facts(None, config.get("model_path")))
    if not slug:
        slug = _match_slug_by_facts(facts)
    if not slug:
        return {"available": True, "matched": False,
                "reason": "该模型不在实测数据库的覆盖范围内"}

    summary = measured_repo.summarise_for_slug(slug)
    if not summary.get("rows"):
        return {"available": True, "matched": False, "slug": slug,
                "reason": summary.get("reason") or "实测库中没有该模型的成功记录"}

    rows = summary["rows"]
    same_gpus = [r for r in rows if r.get("gpu_count") == gpus]
    same_seq = [r for r in (same_gpus or rows) if r.get("cutoff_len") == seq]
    candidates = same_seq or same_gpus or rows
    best = sorted(candidates, key=lambda r: _measured_match_key(r, gpus, seq, mbs or 1))[0]

    estimate_tgs = (outcome.get("summary") or {}).get("tgs")
    measured_tgs = best.get("tokens_per_s")
    deviation = None
    if estimate_tgs and measured_tgs:
        deviation = round((estimate_tgs - measured_tgs) / measured_tgs * 100, 1)

    return {
        "available": True,
        "matched": True,
        "slug": slug,
        "model_names": summary.get("model_names"),
        "source": summary.get("source"),
        "measured_is_lora": True,
        "comparability_note": (
            "实测数据为 LoRA 微调（基座冻结，无基座梯度与优化器状态），"
            "估算为全参训练内核。吞吐量可作量级参照，显存不可直接对比——"
            "LoRA 冻结基座后显存显著低于全参估算属预期现象。"
        ),
        "estimate_tgs": estimate_tgs,
        "measured_tgs": measured_tgs,
        "tgs_deviation_percent": deviation,
        "estimate_peak_gib": outcome.get("peak_memory_gib"),
        "measured_peak_gib": (
            round(best["peak_mem_mib"] / 1024, 2) if best.get("peak_mem_mib") else None
        ),
        "best_match": best,
        "match_quality": _match_quality(best, gpus, seq, mbs or 1),
        "match_bs": best.get("bs"),
        "estimate_bs": mbs,
        "row_count": summary.get("count"),
        "oom_count": summary.get("oom_count"),
        "gpu_counts": summary.get("gpu_counts"),
        "rows": rows[:20],
    }


def _estimate_lora(scenario: TrainingScenario, spec: Any) -> dict[str, Any]:
    """Estimate a LoRA fine-tuning run.

    LoRA has its own resource model (see `services/lora_model.py`), not the
    full-parameter kernel: the base is frozen, so there are no base
    gradients or base optimiser states, while the backward pass still
    traverses the base.
    """
    started = time.time()
    hardware = resolve_hardware(scenario.hardware_id)
    if hardware is None:
        raise ScenarioRejected(f"未知硬件型号：{scenario.hardware_id}")

    slug = scenario.model_slug
    if not slug:
        slug = _catalog_slug_by_name(scenario.model_config_path)
    if not slug:
        raise ScenarioRejected(
            "LoRA 估算需要模型目录中的预设模型",
            [{
                "field": "model_slug",
                "severity": "error",
                "message": "未能确定模型标识",
                "suggestion": "请选择目录中的预设模型（LoRA 系数按模型实测拟合）",
            }],
        )

    calibration = lora_calibration.build_calibration()
    try:
        plans = lora_model.three_plans(
            slug=slug,
            seq_length=scenario.parallel.seq_length,
            lora_rank=scenario.lora_r,
            lora_alpha=scenario.lora_alpha,
            lora_target="all" if not scenario.lora_target_modules else "attn_only",
            ds_config=scenario.lora_ds_config,
            liger=scenario.lora_liger,
            calibration=calibration,
        )
    except lora_model.LoraNotSupported as exc:
        raise ScenarioRejected(str(exc), exc.details)

    measured = measured_comparison(scenario, {}, plans, hardware)

    # The "current configuration" view: estimate the scenario's own card and
    # batch count rather than a searched plan.
    current = None
    try:
        current = lora_model.estimate_lora(
            slug=slug,
            num_gpus=scenario.num_gpus,
            micro_batch_size=scenario.parallel.micro_batch_size,
            seq_length=scenario.parallel.seq_length,
            lora_rank=scenario.lora_r,
            lora_alpha=scenario.lora_alpha,
            lora_target="all" if not scenario.lora_target_modules else "attn_only",
            ds_config=scenario.lora_ds_config,
            liger=scenario.lora_liger,
            calibration=calibration,
        )
    except lora_model.LoraNotSupported:
        current = None

    return {
        "ok": True,
        "action": "lora",
        "mode": scenario.mode.value,
        "mode_label": spec.label,
        "mode_kernel_note": spec.kernel_note,
        "hardware": hardware.to_dict(),
        "measured": measured,
        "plans": {
            "current": current,
            "minimum": plans["minimum"],
            "stable": plans["stable"],
            "performance": plans["performance"],
            "_meta": {
                "stable_label": "推荐稳定方案",
                "stable_reasons": [],
                "evaluations": plans["evaluations"],
                "hbm_gib": plans["hbm_gib"],
                "margin_gib": plans["margin_gib"],
            },
        },
        "issues": [],
        "calibration": lora_calibration.summary(calibration),
        "simulator_version": f"lora-calibration@{calibration.get('generated_at')}",
        "elapsed_ms": round((time.time() - started) * 1000, 1),
    }


def estimate(
    scenario: TrainingScenario,
    *,
    action: str = "search",
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce an estimate for a scenario.

    `action` is one of:
      * ``simulate`` — evaluate the scenario's current parallel layout
      * ``search``   — solve for minimum / stable / performance plans
    """
    started = time.time()

    spec = MODE_SPECS[scenario.mode]
    if not spec.modelled:
        raise UnsupportedModeError(spec.unmodelled_reason)

    if scenario.mode is TrainingMode.SFT_LORA:
        return _estimate_lora(scenario, spec)

    hardware = resolve_hardware(scenario.hardware_id)
    if hardware is None:
        raise ScenarioRejected(f"未知硬件型号：{scenario.hardware_id}")
    if not hardware.supported:
        raise ScenarioRejected(hardware.unsupported_reason)

    # Resolve the config: an explicit path wins, otherwise fall back to the
    # catalogue entry for `model_slug`. Without one of the two there is no
    # way to know the model's shape, so no estimate can be produced.
    config_path = scenario.model_config_path or _catalog_config_path(scenario.model_slug)
    if not config_path:
        raise ScenarioRejected(
            "缺少模型 config.json，无法估算",
            [{
                "field": "model_config_path",
                "severity": "error",
                "message": "既没有提供 config.json 路径，也没有匹配到目录中的预设模型",
                "suggestion": "请在模型选择中指定预设模型，或用 ModelScope / 上传提供 config.json",
            }],
        )

    facts = model_facts(model_config or load_model_facts(scenario.model_slug, config_path))

    issues = validate_scenario(scenario, facts, gpus_per_node=hardware.gpus_per_node)
    if has_errors(issues):
        raise ScenarioRejected(
            "训练参数存在非法组合", [i.to_dict() for i in issues]
        )

    # The simulator needs a path to read the model from; make sure it is the
    # resolved one.
    scenario = replace(scenario, model_config_path=config_path)

    if action == "simulate":
        config = build_simulator_config(
            scenario,
            hardware.id,
            tp_size=scenario.parallel.tp_size,
            pp_size=scenario.parallel.pp_size,
            cp_size=scenario.parallel.cp_size,
            ep_size=scenario.parallel.ep_size,
            etp_size=scenario.parallel.etp_size,
        )
        raw = adapter.simulate(config)
        plans = {"current": adapter.summarise_simulation(raw)}
    else:
        config = build_simulator_config(scenario, hardware.id)
        raw = adapter.solve(config)
        summary = adapter.summarise_solve(raw)
        minimum = summary["minimum"]
        performance = summary["performance"]

        stable_state, stable_reasons = _classify_plan(performance, hardware.hbm_gib)
        if stable_state != "stable":
            # The aggressive plan is not safe to recommend as "stable";
            # fall back to the minimum plan and explain.
            stable = dict(minimum)
            stable_label = "最低可训练方案（性能方案风险过高，已降级）"
        else:
            stable = dict(performance)
            stable_label = "推荐稳定方案"

        plans = {
            "minimum": minimum,
            "stable": stable,
            "performance": performance,
        }
        plans["_meta"] = {
            "stable_label": stable_label,
            "stable_reasons": stable_reasons,
        }

    return {
        "ok": True,
        "action": action,
        "mode": scenario.mode.value,
        "mode_label": spec.label,
        "mode_kernel_note": spec.kernel_note,
        "hardware": hardware.to_dict(),
        "measured": measured_comparison(scenario, facts, plans, hardware),
        "model": facts,
        "plans": plans,
        "issues": [i.to_dict() for i in issues],
        "simulator_version": adapter.simulator_version(),
        "elapsed_ms": round((time.time() - started) * 1000, 1),
    }
