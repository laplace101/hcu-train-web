"""Filesystem locations and simulator bootstrap.

The simulator is imported in-process (not shelled out to the CLI) so that the
web UI consumes the exact same Python objects the `hcu-train-sim` command uses.
That guarantees the numbers shown in the UI are the simulator's numbers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = APP_ROOT.parent

CATALOG_DIR = APP_ROOT / "catalog"
MODELS_DIR = CATALOG_DIR / "models"
CATALOG_FILE = CATALOG_DIR / "catalog.json"
BASELINE_FILE = APP_ROOT / "verification" / "baseline.json"

RUNTIME_DIR = APP_ROOT / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
WORK_DIR = RUNTIME_DIR / "work"

_LEGACY_REPO_CANDIDATES = [
    WORKSPACE_ROOT / "hcu-train-simulator",
    WORKSPACE_ROOT / "hcu-train-simulator-main",
    Path(r"C:\hcu-train-simulator"),
    Path("/root/hcu-train-simulator"),
]


def simulator_src() -> Path:
    """Locate the hcu-train-simulator checkout and return its `src` directory."""

    override = os.environ.get("HCU_SIM_SRC")
    if override:
        candidate = Path(override).expanduser().resolve()
        if not (candidate / "hcu_train_simulator" / "simulator.py").exists():
            raise RuntimeError(
                f"HCU_SIM_SRC={candidate} does not contain hcu_train_simulator/simulator.py"
            )
        return candidate

    for base in _LEGACY_REPO_CANDIDATES:
        src = base / "src"
        if (src / "hcu_train_simulator" / "simulator.py").exists():
            return src

    raise RuntimeError(
        "hcu-train-simulator checkout not found. Clone it next to this project "
        "(expected I:\\train_web\\hcu-train-simulator) or set HCU_SIM_SRC."
    )


_BOOTSTRAPPED = False


def bootstrap() -> Path:
    """Put the simulator on sys.path and import it safely.

    Importing `hcu_train_simulator.simulator` first is required: importing
    `hcu_train_simulator.models.registry` on its own triggers a circular import
    between the qwen_vl and qwen3_5 benchmark modules.
    """

    global _BOOTSTRAPPED
    src = simulator_src()
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    import hcu_train_simulator.simulator  # noqa: F401  (import order matters)

    _BOOTSTRAPPED = True
    return src


def ensure_dirs() -> None:
    for path in (CATALOG_DIR, MODELS_DIR, RUNTIME_DIR, UPLOAD_DIR, WORK_DIR):
        path.mkdir(parents=True, exist_ok=True)
    _prune_work_files()


# Scratch YAML configs are written on every simulation. They are useful while
# debugging but must not grow without bound; anything older than three days is
# removed. Uploaded config.json files are user data and are never touched.
_WORK_RETENTION_DAYS = 3


def _prune_work_files() -> None:
    import time

    cutoff = time.time() - _WORK_RETENTION_DAYS * 86400
    try:
        stale = [p for p in WORK_DIR.glob("*.yaml") if p.stat().st_mtime < cutoff]
    except OSError:
        return
    for path in stale:
        try:
            path.unlink()
        except OSError:
            pass