"""Validate the LoRA model against every measured run.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_lora_model.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repositories import measured as measured_repo  # noqa: E402
from services import lora_calibration, lora_model  # noqa: E402
from services.lora_calibration import _normalise  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def _slug_map() -> dict[str, str]:
    path = PROJECT_ROOT / "catalog" / "catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in data.get("models", []):
        for key in (entry.get("name"), entry.get("hf_id")):
            if key:
                out[_normalise(key)] = entry["slug"]
    return out


def main() -> int:
    cal = lora_calibration.build_calibration(refresh=True)
    fits = cal["step_time"]["fits"]
    print(f"=== 校准 ===")
    print(f"  实测行数: {cal['row_count']}")
    print(f"  step_time 拟合组: {len(fits)}")
    measured_fit = [f for f in fits if f.get("confidence") == "measured_fit"]
    print(f"  其中多样本拟合: {len(measured_fit)}，单样本低置信: {len(fits) - len(measured_fit)}")
    for f in fits:
        mae = f.get("mae_percent")
        mae_s = f"{mae:.1f}%" if mae is not None else "n/a"
        print(f"    {f['slug']:<18} ds={f['ds_config']:<5} lig={f['liger']} n={f['samples']:<3} "
              f"{f.get('confidence', '?'):<20} fixed={f['fixed_s']:>8.3f}s k={f['per_sample_s']:>8.4f}s/bs MAE={mae_s}")

    check("生成了拟合", len(fits) > 0, f"{len(fits)} 组")
    check("每组都带置信度标记", all(f.get("confidence") for f in fits))

    print("\n=== 逐点重现实测 ===")
    conn = measured_repo._connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT model, gpu_count, bs, grad_accum, cutoff_len, ds_config, liger, "
        "step_time_s, tokens_per_s, peak_mem_mib FROM v_train_bench "
        "WHERE status='OK' AND step_time_s IS NOT NULL AND tokens_per_s IS NOT NULL"
    )]
    conn.close()
    slug_by = _slug_map()

    buckets: dict[str, list[tuple[float, float, dict]]] = {}
    for row in rows:
        slug = slug_by.get(_normalise(row["model"]))
        if not slug:
            continue
        est = lora_model.estimate_lora(
            slug=slug, num_gpus=row["gpu_count"], micro_batch_size=row["bs"],
            seq_length=row["cutoff_len"], ds_config=str(row["ds_config"] or "none"),
            liger=int(row["liger"] or 0), calibration=cal,
            # The measured row's own accumulation factor, so the step time
            # being compared is the same one.
            grad_accum=int(row.get("grad_accum") or 1),
        )
        conf = (est.get("throughput_confidence") or {}).get("confidence", "?")
        # Compare against the same reference the model uses: the step time
        # derived from the measured throughput. The stored `step_time_s`
        # column is inconsistent across ingestion runs, which is why the
        # calibration derives its own.
        # Compare against the same reference the model uses: the time of
        # one training step, derived from the measured throughput. The
        # stored `step_time_s` column is inconsistent across ingestion
        # runs, which is why the calibration derives its own.
        #
        # `tokens_per_s` is the per-card throughput, so a micro-step takes
        # bs × seq / tokens_per_s seconds and a training step (grad_accum
        # micro-steps) takes that many times over.
        reference = row["step_time_s"]
        if row.get("tokens_per_s") and row.get("bs") and row.get("cutoff_len"):
            micro_step = (row["bs"] * row["cutoff_len"] / row["tokens_per_s"])
            reference = micro_step * (row.get("grad_accum") or 1)
        step_err = (est["step_time_s"] - reference) / reference * 100
        mem_err = (est["peak_gib"] - row["peak_mem_mib"] / 1024) / (row["peak_mem_mib"] / 1024) * 100
        buckets.setdefault(conf, []).append((step_err, mem_err, row))

    for name, items in sorted(buckets.items()):
        step_errs = [e for e, _, _ in items]
        mem_errs = [m for _, m, _ in items]
        mae = statistics.mean(abs(e) for e in step_errs)
        mem_mae = statistics.mean(abs(e) for e in mem_errs)
        print(f"\n  --- {name} (n={len(items)}) ---")
        print(f"    step_time MAE {mae:.1f}%  max {max(abs(e) for e in step_errs):.1f}%")
        print(f"    peak_mem  MAE {mem_mae:.1f}%  max {max(abs(e) for e in mem_errs):.1f}%")
        if name == "measured_fit":
            check("多样本拟合 step_time 精度 <15%", mae < 15, f"{mae:.1f}%")
            check("多样本拟合显存精度 <5%", mem_mae < 5, f"{mem_mae:.1f}%")
        else:
            print(f"    (低置信组，精度不作硬性要求)")

    print("\n=== LoRA 估算输出 ===")
    est = lora_model.estimate_lora(
        slug="qwen3-5-2b", num_gpus=8, micro_batch_size=4, seq_length=2048,
        lora_rank=8, ds_config="z3", liger=1, calibration=cal)
    for key in ("slug", "num_gpus", "micro_batch_size", "step_time_s", "tokens_per_s",
                "peak_gib", "memory_source", "measured_rows", "adapter_params"):
        print(f"  {key:<20} {est.get(key)}")
    check("给出单步时间", est.get("step_time_s") is not None)
    check("给出吞吐", est.get("tokens_per_s") is not None)
    check("给出显存", est.get("peak_gib") is not None)
    check("给出显存来源", bool(est.get("memory_source")), est.get("memory_source"))
    check("给出吞吐置信度", bool(est.get("throughput_confidence")))
    check("给出适配器参数量", (est.get("adapter_params") or 0) > 0,
          f"{est.get('adapter_params'):,} 个")

    print("\n=== 三类方案 ===")
    plans = lora_model.three_plans(slug="qwen3-5-2b", seq_length=2048, calibration=cal)
    for key in ("minimum", "stable", "performance"):
        plan = plans[key]
        print(f"  {key:<12} {plan['num_gpus']} 卡 bs={plan['micro_batch_size']:<3} "
              f"peak={plan['peak_gib']} GiB  step={plan['step_time_s']}s  tok/s={plan['tokens_per_s']}")
    check("有最低方案", plans["minimum"].get("num_gpus"))
    check("有稳定方案", plans["stable"].get("num_gpus"))
    check("有性能方案", plans["performance"].get("num_gpus"))
    check("最低方案卡数不多于性能方案",
          plans["minimum"]["num_gpus"] <= plans["performance"]["num_gpus"])
    check("性能方案吞吐不低于最低方案",
          (plans["performance"]["tokens_per_s"] or 0) >= (plans["minimum"]["tokens_per_s"] or 0))

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL LORA-MODEL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
