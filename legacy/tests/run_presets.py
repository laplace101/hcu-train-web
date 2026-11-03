#!/usr/bin/env python3
"""Sweep TP/PP across every preset in MODEL_PRESETS and write a CSV.

What it does:
  1. Hits the running backend at http://127.0.0.1:7766 (or the URL in
     $HCU_BACKEND_URL) to enumerate the preset catalog and pull the
     per-preset hardware defaults.
  2. For each preset, sweeps TP in {1, 2, 4, 8} x PP in {1, 2, 4, 8}
     and POSTs /api/predict to the backend. We use:
        task       = full_ft
        hardware   = a100_80g (cheapest to bring up; preset's adapter
                   resolution is independent of the GPU choice)
        micro_bs   = 1
        global_bs  = max(tp * pp * 8, 32)  -- keep DP*1 always >= 4
        seq_length = 4096
     num_gpus is set to tp * pp, so DP=1 and we don't fight the
     "gpus_per_node must divide num_gpus" assertion.
  3. For each successful run, extracts:
        iteration_ms  (from parsed.simulation.iteration_ms)
        tgs           (tokens / sec; from parsed.simulation.tgs)
        mfu           (from parsed.simulation.mfu)
        params_B      (parsed from the simulator's "Model Size: N NB" line)
        tflops        = 6 * params * tokens_per_step / step_time
     and writes a row to a CSV.

Output:
  - the CSV path is --out (default: ./presets_sweep.csv)
  - rows that returned non-zero exit code or that did not include the
    simulation block are still emitted (with empty step-time/tflops)
    so you can see *which* (preset, tp, pp) combinations are infeasible
    in the lab, not just the ones that worked.
  - final summary prints the row count, success rate, and elapsed time.

Usage:
    python3 tests/run_presets.py            # sweep all presets
    python3 tests/run_presets.py --only qwen3_8b qwen3_14b
    python3 tests/run_presets.py --tp 1 2 4 --pp 1 2
    python3 tests/run_presets.py --out sweep.csv
    python3 tests/run_presets.py --backend http://127.0.0.1:7766
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Iterable

# Re-import the same MODEL_PRESETS the backend uses, so the sweep is
# reproducible without depending on the backend being up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backend as _be  # noqa: E402

# Defaults: 4 x 4 sweep
DEFAULT_TP = (1, 2, 4, 8)
DEFAULT_PP = (1, 2, 4, 8)
DEFAULT_HARDWARE = "nvidia_a100"
DEFAULT_TASK = "full_ft"
DEFAULT_SEQ = 4096
DEFAULT_MBS = 1


def _post_json(url: str, payload: dict, timeout: float = 90.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get_json(url: str, timeout: float = 15.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _parse_model_size_b(stdout: str) -> float | None:
    """Pull the simulator's reported 'Model Size: N.B parameters' line."""
    for line in stdout.splitlines():
        line = line.strip()
        # Match "Model Size: 8.1907B parameters" or "Model Size: 744B parameters"
        if line.startswith("Model Size:") and "B parameters" in line:
            try:
                num = line.split("Model Size:")[1].split("B")[0].strip()
                return float(num)
            except (IndexError, ValueError):
                return None
    return None


def _compute_tflops(params_b: float | None, step_ms: float | None,
                    gbs: int, seq_len: int) -> float | None:
    if params_b is None or step_ms is None or step_ms <= 0:
        return None
    tokens = gbs * seq_len
    flops = 6.0 * params_b * 1e9 * tokens  # 6N approximation
    return flops / (step_ms / 1000.0) / 1e12  # -> TFLOPs/s


def sweep(args) -> None:
    presets = list(_be.MODEL_PRESETS)
    if args.only:
        wanted = set(args.only)
        presets = [p for p in presets if p["key"] in wanted]
        if not presets:
            sys.exit("ERROR: --only filters removed all presets")
    print("Sweeping {n} presets, TP={tp}, PP={pp}, total {runs} runs".format(
        n=len(presets), tp=args.tp, pp=args.pp,
        runs=len(presets) * len(args.tp) * len(args.pp)),
        file=sys.stderr)

    rows: list[dict] = []
    total = len(presets) * len(args.tp) * len(args.pp)
    done = 0
    t0 = time.time()
    fail = 0

    for preset in presets:
        key = preset["key"]
        adapter = preset.get("adapter", "?")
        params_rough_b = float(preset.get("rough_params_B") or 0.0)
        for tp in args.tp:
            for pp in args.pp:
                ng = tp * pp
                # gbs must be a multiple of (mbs * tp * pp * dp) = ng (with dp=1).
                # Use the largest gbs that's a multiple of ng and >= 32 so the
                # sweep has enough work to amortize per-step overhead.
                gbs = max(ng, 32)
                # gbs must also be divisible by mbs (=1) -- always true.
                # mbs * ng * dp = gbs  ->  dp = gbs / (mbs * ng)
                payload = {
                    "input_type": "preset",
                    "preset": key,
                    "task": args.task,
                    "hardware": args.hardware,
                    "tp_size": tp,
                    "pp_size": pp,
                    "ep_size": 1,
                    "cp_size": 1,
                    "micro_batch_size": DEFAULT_MBS,
                    "global_batch_size": gbs,
                    "seq_length": args.seq,
                    "num_gpus": ng,
                    "use_fp8": False,
                }
                url = args.backend.rstrip("/") + "/api/predict"
                row = {
                    "adapter": adapter,
                    "preset": key,
                    "tp": tp,
                    "pp": pp,
                    "num_gpus": ng,
                    "gpu_mem_gb": "",  # filled below from hw catalog
                    "params_b_rough": params_rough_b,
                    "est_step_time_s": "",
                    "est_tflops": "",
                    "tgs": "",
                    "mfu": "",
                    "simulator_rc": "",
                    "yaml_path": "",
                    "note": "",
                }
                # Look up GPU mem for the chosen hardware
                try:
                    cat = _get_json(args.backend.rstrip("/") + "/api/presets", timeout=10)
                    hw = next((h for h in cat.get("hardware", [])
                               if h.get("key") == args.hardware), None)
                    if hw:
                        row["gpu_mem_gb"] = hw.get("hbm_gib", "")
                except Exception:
                    pass

                try:
                    r = _post_json(url, payload, timeout=args.timeout)
                except urllib.error.HTTPError as e:
                    row["note"] = "http {} {}".format(e.code, e.reason)
                    fail += 1
                except Exception as e:
                    row["note"] = repr(e)
                    fail += 1
                else:
                    row["simulator_rc"] = r.get("simulator", {}).get("returncode", "")
                    row["yaml_path"] = r.get("yaml_path", "")
                    sim = (r.get("parsed") or {}).get("simulation") or {}
                    if sim.get("iteration_ms"):
                        try:
                            step_ms = float(sim["iteration_ms"])
                            row["est_step_time_s"] = "{:.4f}".format(step_ms / 1000.0)
                            tgs = float(sim.get("tgs", "0") or 0)
                            mfu_raw = sim.get("mfu", "")
                            mfu = float(mfu_raw) if mfu_raw not in (None, "", "?") else 0.0
                            row["tgs"] = "{:.0f}".format(tgs) if tgs else ""
                            row["mfu"] = "{:.4f}".format(mfu) if mfu else ""
                            # Pull the actual params from the simulator's stdout
                            params_actual_b = _parse_model_size_b(
                                r.get("simulator", {}).get("stdout", ""))
                            tf = _compute_tflops(params_actual_b, step_ms, gbs, args.seq)
                            if tf is not None:
                                row["est_tflops"] = "{:.1f}".format(tf)
                        except (ValueError, TypeError):
                            row["note"] = (row.get("note", "") +
                                           " | could not parse iteration_ms").strip(" |")
                    err = r.get("error")
                    if err:
                        row["note"] = (row.get("note", "") + " | " + str(err))[:300]
                        fail += 1
                rows.append(row)
                done += 1
                if done % 5 == 0 or done == total:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    print("  [{d}/{T}] {k} tp={tp} pp={pp} -> rc={rc}, tflops={tf}, "
                          "step_s={st}, note={nt}  (eta {e:.0f}s)".format(
                              d=done, T=total, k=key, tp=tp, pp=pp,
                              rc=row["simulator_rc"], tf=row["est_tflops"],
                              st=row["est_step_time_s"], nt=row["note"][:60],
                              e=eta),
                          file=sys.stderr)

    # Write CSV
    cols = ["adapter", "preset", "tp", "pp", "num_gpus", "gpu_mem_gb",
            "params_b_rough", "est_step_time_s", "est_tflops",
            "tgs", "mfu", "simulator_rc", "yaml_path", "note"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    elapsed = time.time() - t0
    success = sum(1 for r in rows if r.get("simulator_rc") == 0)
    print("\nWrote {n} rows to {p} ({ok} ok, {bad} failed) in {e:.1f}s".format(
        n=len(rows), p=args.out, ok=success, bad=fail, e=elapsed))
    print("Re-run with:  python3 tests/run_presets.py --only <preset>")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default=os.environ.get("HCU_BACKEND_URL",
                                                      "http://127.0.0.1:7766"),
                   help="Backend URL (default: http://127.0.0.1:7766)")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to specific preset keys")
    p.add_argument("--tp", nargs="+", type=int, default=DEFAULT_TP)
    p.add_argument("--pp", nargs="+", type=int, default=DEFAULT_PP)
    p.add_argument("--hardware", default=DEFAULT_HARDWARE)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--out", default="presets_sweep.csv")
    args = p.parse_args()
    sweep(args)


if __name__ == "__main__":
    main()
