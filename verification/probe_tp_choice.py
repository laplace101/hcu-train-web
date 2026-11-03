"""Probe: how does TP choice affect small vs large models?

Uses the same in-process engine the web UI uses, so every number here is
exactly what the UI would show for that configuration.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import catalog as catalog_module  # noqa: E402
from engine import config_builder, engine  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODELS = ["qwen3-0-6b", "qwen3-1-7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"]
TPS = [1, 2, 4, 8]

entries = {m["slug"]: m for m in catalog_module.load_catalog()}


def run(slug, tp, gpus):
    cfg = config_builder.build_config(
        str(ROOT / entries[slug]["config_path"]),
        task="full",
        hardware="bw1000",
        operator_profile="bw1000",
        parallel={
            "tp_size": tp, "pp_size": 1, "ep_size": 1, "cp_size": 1,
            "num_gpus": gpus, "seq_length": 4096,
            "micro_batch_size": 1, "global_batch_size": 256,
        },
    )
    # The simulator is chatty (it prints its own report tables); swallow it.
    with contextlib.redirect_stdout(io.StringIO()):
        out = engine.simulate(cfg, name_hint=f"probe-{slug}")
    s = out["summary"]
    total = s["compute_ms"] + s["communication_ms"]
    return {
        "tp": tp, "gpus": gpus, "dp": gpus // tp,
        "iter": s["iteration_ms"], "tgs": s["tgs"], "mfu": s["mfu"],
        "comm_pct": round(s["communication_ms"] / total * 100, 1) if total else 0.0,
        "mem": out["peak_memory_gib"],
    }


for slug in MODELS:
    e = entries[slug]
    print("=" * 78)
    print(f"{e['name']}  ({e['params_b']}B, adapter={e['adapter']}, "
          f"catalog suggested_tp={e['suggested_tp']}, suggested_gpus={e['suggested_num_gpus']})")
    for label, gpus_fn in (("8卡集群", lambda tp: 8), ("纯TP(卡数=TP)", lambda tp: tp)):
        print(f"\n  --- {label} ---")
        hdr = "%3s %4s %5s %12s %11s %8s %9s %8s" % (
            "TP", "GPUs", "DP", "iter(ms)", "TGS", "MFU", "comm%", "mem(GiB)")
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        rows = []
        for tp in TPS:
            try:
                r = run(slug, tp, gpus_fn(tp))
            except Exception as exc:
                print("  %3d  ERROR %s" % (tp, str(exc)[:70]))
                continue
            mark = "   <== catalog 建议值" if tp == e["suggested_tp"] else ""
            print("  %3d %4d %5d %12.1f %11.1f %7.2f%% %8.1f%% %8.2f%s" % (
                r["tp"], r["gpus"], r["dp"], r["iter"], r["tgs"],
                r["mfu"] * 100, r["comm_pct"], r["mem"], mark))
            rows.append(r)
        if rows:
            best = max(rows, key=lambda r: r["tgs"])
            sug = next((r for r in rows if r["tp"] == e["suggested_tp"]), None)
            verdict = ""
            if sug and best["tp"] != sug["tp"]:
                loss = (sug["tgs"] - best["tgs"]) / best["tgs"] * 100
                verdict = (f"  >>> TGS 最优是 TP{best['tp']}（{best['tgs']:.1f}），"
                           f"catalog 的 TP{sug['tp']} 低 {loss:.0f}%")
            elif sug:
                verdict = f"  >>> catalog 的 TP{sug['tp']} 恰好就是 TGS 最优"
            print(verdict)
    print()