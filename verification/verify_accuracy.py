"""Accuracy gate: prove the web backend returns exactly what hcu-train-sim does.

For every preset model we run the *same* config twice:

  A. through the web backend (`engine.simulate`, in-process)
  B. through the official CLI
     (`python -m hcu_train_simulator.cli run <cfg> --no-report`, a separate
      process that parses the simulator's own stdout tables)

Then we compare compute / communication / iteration / TGS / MFU and the
per-PP-stage memory table. Any difference beyond floating point rounding is a
failure - the UI must not be able to show a number the reference tool would not.

Usage:
    python verification/verify_accuracy.py              # every preset
    python verification/verify_accuracy.py --only qwen3-8b deepseek-v3
    python verification/verify_accuracy.py --refresh    # rewrite baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import catalog as catalog_module  # noqa: E402
from engine import config_builder, engine, paths  # noqa: E402

# Metrics compared with a relative tolerance (the CLI prints rounded values).
TOLERANCE = {
    "compute_ms": 0.01,
    "communication_ms": 0.01,
    "iteration_ms": 0.005,
    "tgs": 0.01,
    "mfu": 0.01,
    "peak_memory_gib": 0.01,
}

SIM_ROW = re.compile(
    r"^\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$",
    re.MULTILINE,
)
MEM_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
    re.MULTILINE,
)


def _section(text: str, title: str) -> str:
    marker = f"=== {title} ==="
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1]
    for other in ("=== ", "\n2026-", "\nWARNING", "\nINFO"):
        idx = tail.find(other, 1)
        if idx != -1:
            tail = tail[:idx]
    return tail


def parse_cli(output: str) -> dict:
    sim = _section(output, "simulation summary")
    rows = SIM_ROW.findall(sim)
    if not rows:
        raise ValueError("CLI 输出中找不到 simulation summary 数据行")
    compute, comm, iteration, tgs, mfu = rows[-1]

    mem_section = _section(output, "memory summary")
    mem_rows = MEM_ROW.findall(mem_section)
    memory = [
        {"pp_rank": int(r[0]), "wgo_gib": float(r[1]), "activation_gib": float(r[2]),
         "total_gib": float(r[3]), "utilization": float(r[4])}
        for r in mem_rows
    ]

    size = re.search(r"Model Size:\s*([\d.]+)B parameters", output)
    return {
        "compute_ms": float(compute),
        "communication_ms": float(comm),
        "iteration_ms": float(iteration),
        "tgs": float(tgs),
        "mfu": float(mfu),
        "peak_memory_gib": max((m["total_gib"] for m in memory), default=0.0),
        "memory": memory,
        "model_size_b": float(size.group(1)) if size else None,
    }


def run_cli(config: dict, workdir: Path) -> dict:
    """Invoke the official CLI in a clean subprocess and parse its output."""

    import yaml

    src = paths.simulator_src()
    workdir.mkdir(parents=True, exist_ok=True)
    cfg_path = workdir / "verify.yaml"
    cfg_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [sys.executable, "-m", "hcu_train_simulator.cli", "run", str(cfg_path), "--no-report"],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise ValueError(f"CLI 返回码 {proc.returncode}: {output[-800:]}")
    return parse_cli(output)


def compare(web: dict, cli: dict) -> list[str]:
    """Return a list of human-readable mismatches (empty == identical)."""

    problems = []
    for key, tolerance in TOLERANCE.items():
        a, b = float(web[key]), float(cli[key])
        denom = max(abs(a), abs(b), 1e-9)
        if abs(a - b) / denom > tolerance:
            problems.append(f"{key}: 网页={a} CLI={b} (偏差 {abs(a - b) / denom:.4%})")

    web_mem = web.get("memory") or []
    cli_mem = cli.get("memory") or []
    if len(web_mem) != len(cli_mem):
        problems.append(f"显存 stage 数量不一致: 网页={len(web_mem)} CLI={len(cli_mem)}")
    else:
        for w, c in zip(web_mem, cli_mem):
            if w["pp_rank"] != c["pp_rank"]:
                problems.append(f"pp_rank 不一致: 网页={w['pp_rank']} CLI={c['pp_rank']}")
            for field in ("wgo_gib", "activation_gib", "total_gib"):
                if abs(w[field] - c[field]) > 0.011:
                    problems.append(
                        f"stage#{w['pp_rank']} {field}: 网页={w[field]} CLI={c[field]}"
                    )
    return problems


def backend_metrics(result: dict) -> dict:
    summary = result["summary"]
    return {
        "compute_ms": summary["compute_ms"],
        "communication_ms": summary["communication_ms"],
        "iteration_ms": summary["iteration_ms"],
        "tgs": summary["tgs"],
        "mfu": summary["mfu"],
        "peak_memory_gib": result["peak_memory_gib"],
        "memory": [
            {"pp_rank": s["pp_rank"], "wgo_gib": s["wgo_gib"],
             "activation_gib": s["activation_gib"], "total_gib": s["total_gib"],
             "utilization": s["utilization"]}
            for s in result["memory"]
        ],
        "model_size_b": summary["model_size_b"],
    }


def build_config_for(entry: dict) -> dict:
    model_path = str(ROOT / entry["config_path"])
    baseline = entry.get("baseline") or {}
    parallel = {
        "tp_size": entry.get("suggested_tp", 1),
        "pp_size": entry.get("suggested_pp", 1),
        "ep_size": entry.get("suggested_ep", 1),
        "num_gpus": entry.get("suggested_num_gpus", 8),
        "seq_length": baseline.get("seq_length", 4096),
        "micro_batch_size": baseline.get("micro_batch_size", 1),
        "global_batch_size": baseline.get("global_batch_size", 256),
    }
    return config_builder.build_config(
        model_path,
        task="full",
        hardware="bw1000",
        operator_profile="bw1000",
        parallel=parallel,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--refresh", action="store_true", help="rewrite baseline.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    models = [m for m in catalog_module.load_catalog() if m.get("verified")]
    if args.only:
        models = [m for m in models if m["slug"] in args.only]
    if args.limit:
        models = models[: args.limit]

    if not models:
        print("没有可校验的预设模型，请先运行 tools/build_catalog.py")
        return 1

    workdir = paths.RUNTIME_DIR / "verify-cli"
    results = []
    failures = []

    print(f"开始校验 {len(models)} 个预设模型：网页后端 vs hcu-train-sim CLI\n")
    for index, entry in enumerate(models, 1):
        slug = entry["slug"]
        config = build_config_for(entry)
        print(f"[{index}/{len(models)}] {slug} ...", end="", flush=True)

        started = time.perf_counter()
        try:
            web = backend_metrics(engine.simulate(config, name_hint=f"acc-{slug}"))
            cli = run_cli(config, workdir / slug)
            problems = compare(web, cli)
        except Exception as exc:  # noqa: BLE001
            print(f" ERROR {type(exc).__name__}: {str(exc)[:200]}")
            failures.append({"slug": slug, "error": f"{type(exc).__name__}: {exc}"})
            continue

        elapsed = round((time.perf_counter() - started) * 1000)
        if problems:
            print(f" MISMATCH ({elapsed} ms)")
            for problem in problems:
                print(f"      - {problem}")
            failures.append({"slug": slug, "problems": problems})
        else:
            print(f" PASS ({elapsed} ms)  iter={cli['iteration_ms']}ms tgs={cli['tgs']} mfu={cli['mfu']}")
            results.append(
                {
                    "slug": slug,
                    "name": entry["name"],
                    "adapter": entry.get("adapter"),
                    "params_b": entry.get("params_b"),
                    "config": {
                        "parallel": config["parallel_config"],
                        "hardware": config["hardware_config"],
                    },
                    "web": web,
                    "cli": cli,
                }
            )

    print(f"\n{'=' * 64}")
    print(f"通过 {len(results)} / {len(models)}，失败 {len(failures)}")
    if failures:
        print("\n失败清单：")
        for failure in failures:
            print(f"  - {failure['slug']}: {failure.get('problems') or failure.get('error')}")

    if args.refresh and results:
        paths.BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        paths.BASELINE_FILE.write_text(
            json.dumps(
                {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "tolerance": TOLERANCE, "verified": results},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\n基线已写入 {paths.BASELINE_FILE}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())