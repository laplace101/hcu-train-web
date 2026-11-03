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
from repositories.hardware import HARDWARE_PROFILES, resolve_hardware
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
        "model": facts,
        "plans": plans,
        "issues": [i.to_dict() for i in issues],
        "simulator_version": adapter.simulator_version(),
        "elapsed_ms": round((time.time() - started) * 1000, 1),
    }
