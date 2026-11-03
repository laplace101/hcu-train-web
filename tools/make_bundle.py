"""Assemble a self-contained deployment bundle for hcu-train-web.

Produces `I:\\train_web\\dist\\hcu-train-web-deploy` containing the app *and* a
vendored copy of hcu-train-simulator, so the target machine needs nothing but a
Python interpreter and network access to a pip index.

    python tools\\make_bundle.py
    python tools\\make_bundle.py --zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DIST = WORKSPACE / "dist"
BUNDLE = DIST / "hcu-train-web-deploy"

# Everything the app needs at runtime. Legacy code and caches are excluded.
APP_ITEMS = [
    "app.py",
    "requirements.txt",
    "README.md",
    "engine",
    "static",
    "templates",
    "catalog",
    "verification",
    "start-web.cmd",
]

EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "legacy", "runtime"}
SIMULATOR_SRC = WORKSPACE / "hcu-train-simulator"

DEPLOY_FILES = ["deploy.sh", "deploy.cmd", "DEPLOY.md"]


def copy_tree(source: Path, target: Path) -> int:
    count = 0
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        if any(part in EXCLUDE_DIRS for part in relative.parts):
            continue
        destination = target / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="store_true", help="also produce a .zip archive")
    parser.add_argument("--force", action="store_true", help="delete an existing bundle first")
    args = parser.parse_args()

    if not (SIMULATOR_SRC / "src" / "hcu_train_simulator" / "simulator.py").exists():
        print(f"[bundle] hcu-train-simulator not found at {SIMULATOR_SRC}")
        return 1

    if BUNDLE.exists():
        if not args.force:
            print(f"[bundle] {BUNDLE} already exists; rerun with --force to rebuild")
            return 1
        shutil.rmtree(BUNDLE)
    DIST.mkdir(parents=True, exist_ok=True)

    print(f"[bundle] assembling -> {BUNDLE}")
    app_target = BUNDLE / "hcu-train-web"
    app_target.mkdir(parents=True, exist_ok=True)

    total = 0
    for item in APP_ITEMS:
        source = ROOT / item
        if not source.exists():
            print(f"[bundle]   skip missing {item}")
            continue
        if source.is_dir():
            total += copy_tree(source, app_target / item)
        else:
            shutil.copy2(source, app_target / item)
            total += 1
        print(f"[bundle]   + hcu-train-web/{item}")

    print(f"[bundle] vendoring hcu-train-simulator ...")
    sim_target = BUNDLE / "hcu-train-simulator"
    sim_target.mkdir(parents=True, exist_ok=True)
    # Only the runtime-relevant parts; .git / docs / tests are not needed.
    for part in ("src", "LICENSE", "NOTICE", "pyproject.toml", "README.md", "train_sim.py"):
        source = SIMULATOR_SRC / part
        if not source.exists():
            continue
        if source.is_dir():
            total += copy_tree(source, sim_target / part)
        else:
            shutil.copy2(source, sim_target / part)
            total += 1

    for name in DEPLOY_FILES:
        source = ROOT / "tools" / name
        if source.exists():
            shutil.copy2(source, BUNDLE / name)
            total += 1
            print(f"[bundle]   + {name}")

    # Fresh runtime folder so the target never inherits local scratch files.
    (app_target / "runtime").mkdir(exist_ok=True)
    (app_target / "runtime" / ".gitkeep").write_text("", encoding="utf-8")

    size_mb = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file()) / 1024 ** 2
    print(f"\n[bundle] {total} files, {size_mb:.1f} MB -> {BUNDLE}")

    if args.zip:
        archive_path = Path(shutil.make_archive(str(BUNDLE), "zip", root_dir=BUNDLE))
        print(f"[bundle] archive: {archive_path} ({archive_path.stat().st_size / 1024 ** 2:.1f} MB)")

    print("\n[bundle] next: copy this folder to the workstation, then run ./deploy.sh (Linux)")
    print("[bundle]       or deploy.cmd (Windows) from inside the folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())