"""In-process wrapper around hcu-train-simulator.

Accuracy contract
-----------------
Every number the UI shows comes from calling the simulator's own estimators.
We never re-derive a metric with our own formula: the aggregation below mirrors
`hcu_train_simulator.simulator.report_data_processing` step for step, so the
dashboard reproduces the simulator's terminal summary exactly.

Process-local state
-------------------
`hcu_train_simulator.context` keeps the active SimulationConfig in a module
global, and `set_config` refuses to run twice without `reset_config()`. The web
server is multi-threaded, so every simulation runs under a single re-entrant
lock and resets the context before and after.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json as _json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from . import config_builder, paths


class SimulationError(RuntimeError):
    """Raised when the simulator rejects a configuration."""

    def __init__(self, message: str, log: str = ""):
        super().__init__(message)
        self.log = log


_LOCK = threading.RLock()


@contextlib.contextmanager
def _simulator_context():
    """Serialize access and clear the process-local config around each run."""

    with _LOCK:
        from hcu_train_simulator.context import reset_config

        reset_config()
        try:
            yield
        finally:
            reset_config()


@contextlib.contextmanager
def _capture_logs(logger_names: list[str]):
    """Collect simulator log output so the UI can show the real reason on failure."""

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    attached = []
    for name in logger_names:
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        attached.append(logger)
    try:
        yield buffer
    finally:
        for logger in attached:
            logger.removeHandler(handler)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _effective_module_time(module: dict) -> float:
    """Mirror simulator.get_effective_module_time."""

    benchmark = module.get("benchmark_time_ms")
    if _is_number(benchmark):
        return float(benchmark)
    forward = module.get("forward_ms", 0) or 0
    backward = module.get("backward_ms", 0) or 0
    return float(forward + backward)


def write_config(config: dict[str, Any], name_hint: str = "run") -> Path:
    paths.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = paths.WORK_DIR / f"{name_hint}-{stamp}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _resolve_model_path(config: dict[str, Any]) -> str:
    model_path = config["model_path"]
    resolved = Path(model_path)
    if not resolved.is_absolute():
        resolved = paths.WORK_DIR / resolved
    if resolved.is_dir():
        resolved = resolved / "config.json"
    return str(resolved.resolve())


def get_config() -> Any:
    from hcu_train_simulator.context import get_config as _get_config

    return _get_config()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(config: dict[str, Any], *, name_hint: str = "run") -> dict[str, Any]:
    paths.bootstrap()
    config_path = write_config(config, name_hint)
    resolved_config = dict(config)
    resolved_config["model_path"] = _resolve_model_path(config)

    from hcu_train_simulator.simulator import run_simulation

    with _simulator_context():
        precheck(config_path)
        with _capture_logs(["hcu_train_simulator"]) as log_buffer:
            try:
                result = run_simulation(str(config_path), generate_report=False)
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI verbatim
                raise SimulationError(f"{type(exc).__name__}: {exc}", log_buffer.getvalue()) from exc
        log_text = log_buffer.getvalue()

    config = _reload_config(resolved_config, config_path)
    return _summarize(result, config, log_text)


def precheck(config_path: Path) -> Any:
    """Build the SimulationConfig and validate it, translating the repo's raw
    assertions into actionable messages.

    The repository asserts several structural constraints inside
    `communication_simulator.configure_parallelism` that otherwise surface as a
    bare `AssertionError: EP*ETP*EDP*PP (0) != total_gpus (8)` with no hint.
    """

    from hcu_train_simulator.config.loader import validate_config
    from hcu_train_simulator.config.models import SimulationConfig
    from hcu_train_simulator.context import get_config, set_config

    try:
        model_config, estimator_config = _load_raw(config_path)
        cfg = SimulationConfig.from_dict(model_config, estimator_config)
        set_config(cfg)

        parallel = cfg.parallel
        experts = cfg.transformer.num_moe_experts
        if experts:
            etp = parallel.etp_size or parallel.tp_size
            if (parallel.ep_size * etp * parallel.pp_size) == 0 or (
                parallel.num_gpus % (parallel.ep_size * etp * parallel.pp_size)
            ):
                raise ValueError(
                    config_builder.explain_ep_illegal(
                        parallel.num_gpus, parallel.tp_size, parallel.pp_size,
                        parallel.ep_size, parallel.etp_size, experts,
                    )
                )
            if experts % parallel.ep_size:
                raise ValueError(
                    f"专家数 {experts} 不能被 EP={parallel.ep_size} 整除，每个 rank 需要持有整数个专家。"
                )
        try:
            validate_config()
        except AssertionError as exc:
            raise ValueError(_translate_assert(str(exc), cfg)) from exc
        return get_config()
    except SimulationError:
        raise
    finally:
        from hcu_train_simulator.context import reset_config

        reset_config()


def _translate_assert(message: str, cfg: Any) -> str:
    parallel = cfg.parallel
    if "num_query_groups" in message:
        return (
            f"{message}。TP={parallel.tp_size} 过大，请改用 KV 组数（{cfg.transformer.num_query_groups}）"
            f"的约数。"
        )
    if "hidden_size" in message:
        return f"{message}。TP 必须整除 hidden_size（{cfg.transformer.hidden_size}）。"
    if "num_attention_heads" in message:
        return f"{message}。TP 必须整除注意力头数（{cfg.transformer.num_attention_heads}）。"
    if "num_layers" in message:
        return (
            f"{message}。PP={parallel.pp_size} 需要整除层数（{cfg.transformer.num_layers}），"
            "或改用「首/末 stage 层数」自定义切分。"
        )
    if "num_gpus" in message:
        return f"{message}。当前 num_gpus={parallel.num_gpus}，TP×CP×PP={parallel.tp_size * parallel.cp_size * parallel.pp_size}。"
    if "pp_schedule" in message or "num_layers_per_vp_stage" in message:
        return f"{message}。VPP（interleaved 调度）要求 PP>1 且每 PP stage 层数可被整除。"
    return message


def _load_raw(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import json as _json

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["_config_dir"] = str(config_path.parent)
    model_path = Path(data["model_path"])
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    if model_path.is_dir():
        model_path = model_path / "config.json"
    return _json.loads(model_path.read_text(encoding="utf-8")), data


def _reload_config(resolved_config: dict[str, Any], config_path: Path) -> Any:
    """Re-create the SimulationConfig object the run used, for derived lookups.

    Called after `_simulator_context` has cleared the process-local config, so
    `set_config` is safe here. The reloaded object is what `_summarize` reads
    for DP/VP sizes and hardware facts.
    """

    from hcu_train_simulator.config.loader import load_config
    from hcu_train_simulator.config.models import SimulationConfig
    from hcu_train_simulator.context import reset_config, set_config

    reset_config()
    set_config(SimulationConfig.from_dict(*load_config(config_path)))
    return get_config()


def _summarize(result: dict[str, Any], config: Any, log_text: str) -> dict[str, Any]:
    """Mirror report_data_processing so the UI matches the CLI summary."""

    from hcu_train_simulator.context import get_config
    from hcu_train_simulator.estimators.communication import expert_data_parallel_size
    from hcu_train_simulator.modeling import ModelStatistics
    from hcu_train_simulator.modeling.spec import spec_uses_moe

    cfg = get_config()
    compute_result = result["compute"]
    comm_result = result["communication"]
    memory_result = result["memory"]

    model_size = ModelStatistics().compute_model_size()
    model_flops = ModelStatistics().compute_flops()

    all_compute_time = round(sum(_effective_module_time(m) for m in compute_result), 2)
    all_comm_time = round(comm_result["total"] * 1000, 2)
    iteration_ms = round(all_compute_time + all_comm_time, 2)

    parallel = cfg.parallel
    tokens_per_iter = parallel.global_batch_size * parallel.seq_length
    iteration_s = iteration_ms / 1000
    tgs = round(tokens_per_iter / parallel.num_gpus / iteration_s, 2) if iteration_s else 0.0

    actual_tflops = cfg.hardware.fp8_tflops if parallel.use_fp8_training else cfg.hardware.fp16_tflops
    mfu = round(model_flops / (iteration_s * 1e12 * parallel.num_gpus) / actual_tflops, 4)

    raw_comm = comm_result.get("raw", {}) or {}
    overlap_ratios = comm_result.get("overlap_ratios", {}) or {}
    is_moe = spec_uses_moe(cfg.model_spec)
    domains = ["tp", "cp", "ep", "dp", "edp", "pp", "etp"] if is_moe else ["tp", "cp", "dp", "pp"]

    comm_domains = []
    for domain in domains:
        total = comm_result.get(domain, 0) or 0
        comm_domains.append(
            {
                "domain": domain.upper(),
                "raw_time_ms": round((raw_comm.get(domain, total) or 0) * 1000, 2),
                "overlap_percent": round((overlap_ratios.get(domain, 0.0) or 0.0) * 100, 2),
                "time_ms": round(total * 1000, 2),
                "percent": round(total / comm_result["total"] * 100, 2) if comm_result["total"] else 0.0,
                "size": getattr(parallel, f"{domain}_size", None)
                if domain != "edp"
                else expert_data_parallel_size(cfg),
            }
        )

    hbm = cfg.hardware.hbm_gib
    memory_summary = [
        {
            "pp_rank": stage["pp_rank"],
            "wgo_gib": round(stage["wgo_gib"], 2),
            "activation_gib": round(stage["act_gib"], 2),
            "total_gib": round(stage["total_gib"], 2),
            "hbm_gib": hbm,
            "utilization": round(stage["total_gib"] / hbm * 100, 2),
            "feasible": stage["total_gib"] <= hbm,
        }
        for stage in memory_result
    ]

    compute_details = [
        {
            "model_part": module.get("model_part"),
            "op_type": module.get("op_type"),
            "count": module.get("compute_count"),
            "time_ms": round(_effective_module_time(module), 4),
            "forward_ms": module.get("forward_ms"),
            "backward_ms": module.get("backward_ms"),
            "benchmark_source": module.get("benchmark_source"),
            "flops": module.get("flops"),
        }
        for module in compute_result
    ]

    return {
        "summary": {
            "model_size_b": round(model_size / 1e9, 4),
            "model_params": model_size,
            "model_flops": model_flops,
            "compute_ms": all_compute_time,
            "communication_ms": all_comm_time,
            "iteration_ms": iteration_ms,
            "tgs": tgs,
            "mfu": mfu,
            "mfu_percent": round(mfu * 100, 2),
            "comm_percent": round(all_comm_time / iteration_ms * 100, 2) if iteration_ms else 0.0,
            "compute_percent": round(all_compute_time / iteration_ms * 100, 2) if iteration_ms else 0.0,
            "tokens_per_iteration": tokens_per_iter,
            "actual_tflops": actual_tflops,
        },
        "parallel": {
            "num_gpus": parallel.num_gpus,
            "tp_size": parallel.tp_size,
            "cp_size": parallel.cp_size,
            "pp_size": parallel.pp_size,
            "dp_size": parallel.dp_size,
            "ep_size": parallel.ep_size,
            "etp_size": parallel.etp_size,
            "edp_size": expert_data_parallel_size(cfg),
            "vp_size": max(parallel.vp_size, 1),
            "num_layers_per_vp_stage": parallel.num_layers_per_vp_stage,
            "pp_schedule": parallel.pp_schedule,
            "seq_length": parallel.seq_length,
            "micro_batch_size": parallel.micro_batch_size,
            "global_batch_size": parallel.global_batch_size,
            "use_fp8_training": parallel.use_fp8_training,
            "full_recompute": parallel.full_recompute,
            "overlap_mode": parallel.overlap_mode,
            "num_microbatches": config_builder.describe_derived(
                {
                    "tp_size": parallel.tp_size,
                    "cp_size": parallel.cp_size,
                    "pp_size": parallel.pp_size,
                    "num_gpus": parallel.num_gpus,
                    "micro_batch_size": parallel.micro_batch_size,
                    "global_batch_size": parallel.global_batch_size,
                }
            )["num_microbatches"],
        },
        "memory": memory_summary,
        "peak_memory_gib": max((s["total_gib"] for s in memory_summary), default=0.0),
        "hbm_gib": hbm,
        "communication": comm_domains,
        "compute_details": compute_details,
        "model_adapter": cfg.model_adapter,
        "log": log_text,
    }


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(config: dict[str, Any], *, options: dict[str, Any] | None = None) -> dict[str, Any]:
    paths.bootstrap()
    config_path = write_config(config, "solve")
    resolved = dict(config)
    resolved["model_path"] = _resolve_model_path(config)
    _rewrite_model_path(config_path, resolved["model_path"])

    from hcu_train_simulator.commands.solve import SolveOptions, solve_training_config

    opts = options or {}
    solve_options = SolveOptions(
        seq_length=int(opts.get("seq_length") or config["parallel_config"]["seq_length"]),
        micro_batch_size=int(opts.get("micro_batch_size") or config["parallel_config"]["micro_batch_size"]),
        global_batch_size=opts.get("global_batch_size"),
        max_bubble_ratio=float(opts.get("max_bubble_ratio", 0.15)),
        memory_margin_gib=opts.get("memory_margin_gib"),
    )

    with _simulator_context():
        precheck(config_path)
        with _capture_logs(["hcu_train_simulator"]) as log_buffer:
            try:
                result = solve_training_config(str(config_path), solve_options)
            except Exception as exc:  # noqa: BLE001
                raise SimulationError(f"{type(exc).__name__}: {exc}", log_buffer.getvalue()) from exc

    return {
        "minimum": _solve_row(result.get("minimum")),
        "performance": _solve_row(result.get("performance")),
        "assumptions": result.get("assumptions") or [],
        "log": log_buffer.getvalue(),
    }


def _rewrite_model_path(config_path: Path, model_path: str) -> None:
    """solve_training_config resolves relative model paths against the YAML dir."""

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["model_path"] = model_path
    config_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _solve_row(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    parallel = result.get("parallel", {})
    memory = result.get("memory") or []
    peak = max(memory, key=lambda s: s["total_gib"]) if memory else None
    world = parallel.get("num_gpus", 0)
    model_parallel = (
        parallel.get("tp_size", 1) * parallel.get("cp_size", 1) * parallel.get("pp_size", 1)
    )
    return {
        "parallel": {
            "num_gpus": world,
            "nodes": None,
            "tp_size": parallel.get("tp_size"),
            "cp_size": parallel.get("cp_size"),
            "pp_size": parallel.get("pp_size"),
            "dp_size": world // model_parallel if model_parallel else 0,
            "ep_size": parallel.get("ep_size"),
            "etp_size": parallel.get("etp_size"),
            "vp_size": parallel.get("vp_size") if parallel.get("num_layers_per_vp_stage") else None,
            "num_layers_per_vp_stage": parallel.get("num_layers_per_vp_stage"),
            "pp_schedule": parallel.get("pp_schedule", "1f1b"),
            "micro_batch_size": parallel.get("micro_batch_size"),
            "global_batch_size": parallel.get("global_batch_size"),
            "ep_overlap_enabled": parallel.get("ep_overlap_enabled", False),
        },
        "memory": [
            {
                "pp_rank": stage["pp_rank"],
                "wgo_gib": round(stage["wgo_gib"], 2),
                "activation_gib": round(stage["act_gib"], 2),
                "total_gib": round(stage["total_gib"], 2),
            }
            for stage in memory
        ],
        "peak_memory_gib": round(peak["total_gib"], 2) if peak else None,
        "peak_pp_rank": peak["pp_rank"] if peak else None,
        "performance": {
            "iteration_ms": round(result["iteration_time_ms"], 2) if "iteration_time_ms" in result else None,
            "mfu": round(result["mfu"], 4) if "mfu" in result else None,
            "tgs": round(result["tgs"], 2) if "tgs" in result else None,
        },
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def search(config: dict[str, Any], *, top_k: int = 10) -> dict[str, Any]:
    """Grid-search parallel strategies, ranked by TGS.

    The shipped `hcu-train-sim search` evaluates candidates with theoretical
    compute only, so its TGS cannot be compared with the main dashboard (which
    uses measured operator profiles). We therefore reuse the repository's
    estimator sequence from `simulator.run_simulation` verbatim - same
    validation, same profile application, same bubble accounting - so a
    candidate ranked #1 here is consistent with the numbers the user sees when
    they run that exact configuration.
    """

    paths.bootstrap()
    import contextlib
    import copy
    import io
    import json as _json

    from hcu_train_simulator.benchmarks import (
        OperatorProfileCatalog,
        apply_profile_to_compute_result,
        build_benchmark_config,
    )
    from hcu_train_simulator.commands.search import (
        _iter_search_candidates,
        _skip_unneeded_dense_candidate,
    )
    from hcu_train_simulator.config.loader import validate_config
    from hcu_train_simulator.config.models import SimulationConfig
    from hcu_train_simulator.context import reset_config, set_config
    from hcu_train_simulator.estimators import estimate_communication, estimate_compute, estimate_memory
    from hcu_train_simulator.modeling import ModelStatistics
    from hcu_train_simulator.simulator import (
        apply_measured_coverage,
        estimate_pp_schedule_bubble_time_ms,
        fill_missing_benchmark_result,
        get_effective_module_time,
    )

    config_path = write_config(config, "search")
    resolved = dict(config)
    resolved["model_path"] = _resolve_model_path(config)
    _rewrite_model_path(config_path, resolved["model_path"])

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = _json.loads(Path(resolved["model_path"]).read_text(encoding="utf-8"))
    estimator_config = {**data, "_config_dir": str(config_path.parent)}

    candidates: dict[str, Any] = data.get("search", {}).get("candidates", {}) or {}
    base_parallel = dict(data["parallel_config"])

    combos: list[dict[str, Any]] = []
    for combo in _iter_search_candidates(model_config, base_parallel, candidates):
        if _skip_unneeded_dense_candidate(model_config, combo):
            continue
        combos.append(combo)

    rows: list[dict[str, Any]] = []
    margin = float(data.get("search", {}).get("memory_margin_gib", 0) or 0)
    memory_limit = float(data["hardware_config"]["hbm_gib"]) - margin

    with _simulator_context():
        for combo in combos:
            try:
                candidate_config = copy.deepcopy(estimator_config)
                candidate_config["parallel_config"].update(combo)
                if combo.get("num_layers_per_vp_stage"):
                    candidate_config["parallel_config"]["pp_schedule"] = combo.get(
                        "pp_schedule", "interleaved_1f1b"
                    )
                outcome = _evaluate_with_profile(
                    model_config,
                    candidate_config,
                    validate_config=validate_config,
                    estimate_memory=estimate_memory,
                    estimate_compute=estimate_compute,
                    estimate_communication=estimate_communication,
                    ModelStatistics=ModelStatistics,
                    OperatorProfileCatalog=OperatorProfileCatalog,
                    apply_profile_to_compute_result=apply_profile_to_compute_result,
                    build_benchmark_config=build_benchmark_config,
                    apply_measured_coverage=apply_measured_coverage,
                    fill_missing_benchmark_result=fill_missing_benchmark_result,
                    estimate_pp_schedule_bubble_time_ms=estimate_pp_schedule_bubble_time_ms,
                    get_effective_module_time=get_effective_module_time,
                    redirect_stdout=contextlib.redirect_stdout,
                    stringio=io.StringIO,
                    reset=reset_config,
                    set_config=set_config,
                )
            except Exception as exc:  # noqa: BLE001 - infeasible combos are expected
                rows.append(
                    {"combo": combo, "feasible": False,
                     "reason": f"{type(exc).__name__}: {exc}"}
                )
                continue
            finally:
                reset_config()

            peak = max((s["total_gib"] for s in outcome["memory"]), default=0.0)
            if peak > memory_limit:
                rows.append(
                    {"combo": combo, "feasible": False,
                     "reason": f"峰值显存 {peak:.2f} GiB 超过上限 {memory_limit:.2f} GiB"}
                )
                continue
            parallel = outcome["parallel"]
            world = parallel["num_gpus"]
            dp = world // (parallel["tp_size"] * parallel["cp_size"] * parallel["pp_size"])
            rows.append(
                {
                    "combo": combo,
                    "feasible": True,
                    "num_gpus": world,
                    "tp_size": parallel["tp_size"],
                    "cp_size": parallel["cp_size"],
                    "pp_size": parallel["pp_size"],
                    "dp_size": dp,
                    "ep_size": parallel["ep_size"],
                    "etp_size": parallel["etp_size"],
                    "num_layers_per_vp_stage": parallel.get("num_layers_per_vp_stage"),
                    "vp_size": parallel.get("vp_size") if parallel.get("num_layers_per_vp_stage") else None,
                    "pp_schedule": parallel.get("pp_schedule", "1f1b"),
                    "micro_batch_size": parallel["micro_batch_size"],
                    "peak_memory_gib": round(peak, 2),
                    "iteration_ms": round(outcome["iteration_time_ms"], 2),
                    "tgs": round(outcome["tgs"], 2),
                    "mfu": round(outcome["mfu"], 4),
                }
            )

    feasible = [r for r in rows if r["feasible"]]
    feasible.sort(key=lambda r: r["tgs"], reverse=True)
    return {
        "evaluated": len(rows),
        "feasible": len(feasible),
        "top": feasible[:top_k],
        "rejected": [r for r in rows if not r["feasible"]][:20],
        "memory_limit_gib": memory_limit,
        "objective": "tgs",
    }


def _evaluate_with_profile(model_config, candidate_config, **deps):
    """Evaluate one candidate using run_simulation's estimator sequence."""

    reset = deps["reset"]
    set_config = deps["set_config"]
    reset()
    try:
        from hcu_train_simulator.config.models import SimulationConfig

        cfg = SimulationConfig.from_dict(model_config, candidate_config)
        set_config(cfg)

        with deps["redirect_stdout"](deps["stringio"]()):
            deps["validate_config"]()
            memory_result = deps["estimate_memory"]()
            compute_result = deps["estimate_compute"]()

        benchmark_config = deps["build_benchmark_config"](cfg)
        profile_catalog = deps["OperatorProfileCatalog"](cfg)
        mode = cfg.profile.mode if cfg.profile.enabled else "theoretical"
        if mode == "builtin" and profile_catalog.select_builtin():
            compute_result = deps["apply_profile_to_compute_result"](
                compute_result, profile_catalog, cfg
            )
            compute_result = deps["apply_measured_coverage"](compute_result)
        compute_result = deps["fill_missing_benchmark_result"](compute_result)

        comm_result = deps["estimate_communication"](compute_result)
        bubble_ms = deps["estimate_pp_schedule_bubble_time_ms"](compute_result, cfg)
        comm_result["pp"] = comm_result.get("pp", 0.0) + bubble_ms / 1000
        comm_result["total"] = comm_result.get("total", 0.0) + bubble_ms / 1000

        compute_ms = sum(deps["get_effective_module_time"](m) for m in compute_result)
        comm_ms = comm_result["total"] * 1000
        iteration_ms = compute_ms + comm_ms
        tgs = (
            cfg.parallel.global_batch_size
            * cfg.parallel.seq_length
            / cfg.parallel.num_gpus
            / (iteration_ms / 1000)
        )
        model_flops = deps["ModelStatistics"]().compute_flops()
        actual_tflops = cfg.hardware.fp8_tflops if cfg.parallel.use_fp8_training else cfg.hardware.fp16_tflops
        mfu = model_flops / (iteration_ms / 1000 * 1e12 * cfg.parallel.num_gpus) / actual_tflops

        parallel = dict(candidate_config["parallel_config"])
        parallel["vp_size"] = max(cfg.parallel.vp_size, 1)
        parallel["num_gpus"] = cfg.parallel.num_gpus
        return {
            "parallel": parallel,
            "memory": memory_result,
            "iteration_time_ms": iteration_ms,
            "tgs": tgs,
            "mfu": mfu,
        }
    finally:
        reset()