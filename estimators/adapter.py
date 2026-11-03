"""Estimator adapter (plan §5.2, B05).

This is the ONLY place that talks to `hcu-train-simulator`. Everything above
it works with plain dicts.

Two problems this layer solves:

1. **Process-global state.** The simulator keeps its configuration in a
   module-level global (`context._CONFIG`) and `set_config` raises when the
   context was not reset. Calling it twice in one process therefore leaks
   state between calls. The existing `engine/engine.py` serialises every
   simulation behind one global lock with `reset_config()` on both sides.
   That works for a single-process web app but scales badly, and the plan
   explicitly asks for "任务进程隔离或受控单任务 worker" (B05).

   This adapter runs each estimate in a **short-lived subprocess**, so
   concurrent requests cannot interfere. The subprocess reports back over a
   JSON marker on stdout.

2. **Aggregation drift.** The simulator's `run_simulation` returns raw
   per-operator rows, not the summary numbers the CLI prints. Re-deriving
   iteration time / TGS / MFU in the web layer would be a second
   implementation of the formulas — exactly what §5.3 forbids. The
   subprocess therefore reuses the simulator's own aggregation helpers
   (`get_effective_module_time`, `ModelStatistics`), so the numbers are the
   simulator's numbers by construction.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = APP_ROOT / "runtime" / "work"

#: Marker used to separate the JSON payload from the simulator's own logging.
_RESULT_MARKER = "@@ESTIMATE_RESULT@@"


class EstimationError(RuntimeError):
    """Raised when an estimate could not be produced."""


class UnsupportedModeError(EstimationError):
    """Raised for a training mode the estimator kernel does not model."""


#: Script executed in the isolated child process. It reuses the project's
#: own `engine` module, so the numbers come from exactly the same code path
#: as the legacy web routes and the 35/35 accuracy baseline.
_RUNNER = r'''
"""Isolated estimate runner (written to runtime/work and executed as a child).

Usage: python _estimate_runner.py <config.yaml> <simulate|solve>

Prints the JSON payload after the marker line so the parent can separate it
from the simulator's own logging.
"""
import json
import os
import sys

sys.path.insert(0, os.environ["SIM_SRC"])
sys.path.insert(0, os.environ["APP_ROOT"])

from engine import engine  # noqa: E402


def main():
    config_path, mode = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.dirname(os.path.abspath(config_path)))

    import yaml
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if mode == "solve":
        options = config.get("search") or {}
        result = engine.solve(config, options=options)
    else:
        result = engine.simulate(config)

    marker = os.environ["RESULT_MARKER"]
    sys.stdout.write(marker)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
'''


def _sim_src() -> Path | None:
    """Locate the simulator's `src` directory."""
    override = os.environ.get("HCU_SIM_SRC")
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates += [
        APP_ROOT.parent / "hcu-train-simulator" / "src",
        Path("/root/hcu-train-simulator/src"),
        Path("/root/hygon/train_web/hcu-train-simulator/src"),
    ]
    for candidate in candidates:
        if (candidate / "hcu_train_simulator" / "simulator.py").exists():
            return candidate
    return None


def simulator_available() -> bool:
    return _sim_src() is not None


def simulator_version() -> str:
    """Version string of the simulator checkout.

    The checkout may be a git clone, a source archive, or a plain directory,
    so several sources are tried in turn: the git HEAD, the project version
    in `pyproject.toml`, and finally the mtime of the simulator package.
    """
    src = _sim_src()
    if src is None:
        return "unknown"
    repo = src.parent

    # 1. git checkout
    head = repo / ".git" / "HEAD"
    if head.exists():
        try:
            ref = head.read_text(encoding="utf-8").strip()
            if ref.startswith("ref:"):
                sha_file = repo / ".git" / ref.split(" ", 1)[1].strip()
                if sha_file.exists():
                    return sha_file.read_text(encoding="utf-8").strip()[:12]
            elif ref:
                return ref[:12]
        except OSError:
            pass

    # 2. project version
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        try:
            import re

            text = pyproject.read_text(encoding="utf-8")
            match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
            if match:
                return f"{match.group(1)} (source)"
        except OSError:
            pass

    # 3. last-modified marker of the simulator package
    package = src / "hcu_train_simulator"
    if package.exists():
        newest = max(
            (p.stat().st_mtime for p in package.rglob("*.py")),
            default=None,
        )
        if newest:
            import datetime

            stamp = datetime.datetime.fromtimestamp(newest).strftime("%Y-%m-%d")
            return f"source@{stamp}"
    return "unknown"


def _run_in_subprocess(config: dict[str, Any], mode: str, timeout: int = 300) -> dict[str, Any]:
    """Run the simulator in an isolated process and return its JSON payload."""
    src = _sim_src()
    if src is None:
        raise EstimationError(
            "未找到 hcu-train-simulator 源码。请将仓库放在项目同级目录，"
            "或设置环境变量 HCU_SIM_SRC 指向其 src 目录。"
        )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    import yaml

    config_path = WORK_DIR / f"estimate-{mode}-{os.getpid()}.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    runner_path = WORK_DIR / "_estimate_runner.py"
    runner_path.write_text(_RUNNER, encoding="utf-8")

    env = dict(os.environ)
    env["SIM_SRC"] = str(src)
    env["APP_ROOT"] = str(APP_ROOT)
    env["RESULT_MARKER"] = _RESULT_MARKER

    try:
        proc = subprocess.run(
            [sys.executable, str(runner_path), str(config_path), mode],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORK_DIR),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise EstimationError(f"估算超时（{timeout} 秒）") from exc
    finally:
        config_path.unlink(missing_ok=True)

    if _RESULT_MARKER in proc.stdout:
        payload = proc.stdout.split(_RESULT_MARKER, 1)[1].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EstimationError("估算结果解析失败") from exc

    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    tail = detail[-1] if detail else "未知错误"
    raise EstimationError(f"估算执行失败：{tail[:300]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate(config: dict[str, Any], *, timeout: int = 300) -> dict[str, Any]:
    """Run one simulation of a fixed configuration."""
    return _run_in_subprocess(config, "simulate", timeout=timeout)


def solve(config: dict[str, Any], *, timeout: int = 600) -> dict[str, Any]:
    """Search for the minimum-resource and higher-performance plans."""
    return _run_in_subprocess(config, "solve", timeout=timeout)


def summarise_solution(solution: Any) -> dict[str, Any]:
    """Flatten one solve solution into render-ready fields.

    The input is `engine.solve`'s row shape:
        {parallel: {...}, memory: [...], peak_memory_gib, performance: {iteration_ms, mfu, tgs}}
    """
    if not isinstance(solution, dict):
        return {}

    parallel = solution.get("parallel") or {}
    memory_rows = solution.get("memory") or []
    perf = solution.get("performance") or {}

    peak = max(memory_rows, key=lambda r: r.get("total_gib", 0), default=None)

    out: dict[str, Any] = {
        "num_gpus": parallel.get("num_gpus"),
        "tp_size": parallel.get("tp_size"),
        "pp_size": parallel.get("pp_size"),
        "cp_size": parallel.get("cp_size"),
        "dp_size": parallel.get("dp_size"),
        "ep_size": parallel.get("ep_size"),
        "etp_size": parallel.get("etp_size"),
        "vp_size": parallel.get("vp_size"),
        "micro_batch_size": parallel.get("micro_batch_size"),
        "global_batch_size": parallel.get("global_batch_size"),
        "seq_length": parallel.get("seq_length"),
        "pp_schedule": parallel.get("pp_schedule"),
        "peak_gib": solution.get("peak_memory_gib"),
        "wgo_gib": round(peak["wgo_gib"], 2) if peak else None,
        "activation_gib": round(peak["activation_gib"], 2) if peak else None,
        "iteration_ms": perf.get("iteration_ms"),
        "mfu": perf.get("mfu"),
        "tgs": perf.get("tgs"),
    }
    return {k: v for k, v in out.items() if v is not None}


def summarise_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten `engine.simulate`'s payload into the fields the UI shows."""
    summary = payload.get("summary") or {}
    parallel = payload.get("parallel") or {}
    memory = payload.get("memory") or []
    peak = max(memory, key=lambda r: r.get("total_gib", 0), default=None)

    return {
        "iteration_ms": summary.get("iteration_ms"),
        "compute_time_ms": summary.get("compute_ms"),
        "communication_time_ms": summary.get("communication_ms"),
        "mfu": summary.get("mfu"),
        "tgs": summary.get("tgs"),
        "peak_gib": payload.get("peak_memory_gib"),
        "wgo_gib": round(peak["wgo_gib"], 2) if peak else None,
        "activation_gib": round(peak["activation_gib"], 2) if peak else None,
        "num_gpus": parallel.get("num_gpus"),
        "tp_size": parallel.get("tp_size"),
        "pp_size": parallel.get("pp_size"),
        "cp_size": parallel.get("cp_size"),
        "ep_size": parallel.get("ep_size"),
        "dp_size": parallel.get("dp_size"),
        "hbm_gib": payload.get("hbm_gib"),
        "model_adapter": payload.get("model_adapter"),
    }


def summarise_solve(result: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw solve payload into the three plans the UI shows."""
    return {
        "minimum": summarise_solution(result.get("minimum")),
        "performance": summarise_solution(result.get("performance")),
        "assumptions": result.get("assumptions") or [],
    }
