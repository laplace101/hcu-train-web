"""Checks that LoRA and full-parameter measurements never mix.

The measured database holds both kinds of run, told apart by
``finetuning_type``. They measure different training regimes: LoRA freezes
the base weights, so it carries no base gradients and no base optimiser
state. A full-parameter row therefore says nothing about whether a LoRA
configuration fits in memory, and a LoRA row says nothing about whether a
full-parameter configuration does.

Mixing them has a concrete consequence: a full-parameter run that ran out
of memory would mark the *same* LoRA configuration as impossible, even
though the LoRA campaign ran it successfully. This test locks that down.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_measured_modes.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repositories import measured as measured_repo  # noqa: E402
from services import lora_calibration as lc  # noqa: E402
from services import lora_model  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def _kinds_of_run(row: dict) -> set[str]:
    """The kind of run a measured row came from.

    The calibration rows carry their own `finetuning_type`, so the answer
    is read from the row rather than looked up by configuration — a
    configuration can have been run by both campaigns, and matching on it
    would report a false positive.
    """
    kind = (row.get("finetuning_type") or "").strip().lower()
    return {kind} if kind else set()


def main() -> int:
    print("=== ① 过滤器把两类运行分开 ===")
    conn = sqlite3.connect(
        f"file:{measured_repo.database_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    for mode, allowed in (("lora", {"lora", "qlora"}), ("full", {"full"})):
        counts: dict[str, int] = {}
        for row in conn.execute(
            "SELECT finetuning_type, COUNT(*) n FROM train_bench WHERE 1=1"
            + measured_repo._finetuning_filter(mode)
            + " GROUP BY finetuning_type"
        ):
            counts[row["finetuning_type"] or "(null)"] = row["n"]
        check(f"{mode} 过滤器只选出该类运行",
              all(k in allowed for k in counts), str(counts))

    lora_n = conn.execute(
        "SELECT COUNT(*) FROM train_bench WHERE 1=1"
        + measured_repo._finetuning_filter("lora")
    ).fetchone()[0]
    full_n = conn.execute(
        "SELECT COUNT(*) FROM train_bench WHERE 1=1"
        + measured_repo._finetuning_filter("full")
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM train_bench").fetchone()[0]
    check("两个过滤器正好划分整张表", lora_n + full_n == total,
          f"{lora_n} + {full_n} = {total}")

    print("\n=== ② LoRA 校准只读 LoRA 行 ===")
    rows = lc._measured_rows()
    kinds: set[str] = set()
    for row in rows:
        kinds |= _kinds_of_run(row)
    check("校准行不含全参运行", kinds <= {"lora", "qlora"}, str(kinds))
    check("校准行不为空", len(rows) > 0, f"{len(rows)} 行")

    oom_rows = lc._measured_oom_rows()
    oom_kinds: set[str] = set()
    for row in oom_rows:
        oom_kinds |= _kinds_of_run(row)
    check("OOM 边界不含全参运行", oom_kinds <= {"lora", "qlora"}, str(oom_kinds))

    print("\n=== ③ LoRA 显存分组只读 LoRA 行 ===")
    group = lora_model._measured_group("qwen3-5-2b", "z3", 1)
    check("显存分组非空", len(group) > 0, f"{len(group)} 行")
    # The full-parameter run at this configuration peaked at 10597 MiB.
    # LoRA does not carry the base gradients and optimiser state, so its
    # peak at the same configuration is far lower. If the two are mixed,
    # this number appears.
    peaks = [r.get("peak_mem_mib") for r in group]
    check("显存分组不含全参峰值", 10597 not in peaks,
          f"峰值 {sorted({p for p in peaks if p})}")

    print("\n=== ④ 全参 OOM 不得把 LoRA 配置判为不可行 ===")
    # Find a configuration where the LoRA campaign succeeded and the
    # full-parameter campaign ran out of memory. The LoRA estimate must
    # not claim that configuration ran out of memory.
    conflicting = []
    for row in conn.execute(
        "SELECT l.model, l.bs, l.cutoff_len, l.ds_config, l.liger, l.gpu_count "
        "FROM v_train_bench l "
        "WHERE l.status='OK' AND l.finetuning_type='lora' "
        "  AND EXISTS (SELECT 1 FROM train_bench f "
        "              WHERE f.model=l.model AND f.bs=l.bs "
        "                AND f.cutoff_len=l.cutoff_len "
        "                AND f.ds_config=l.ds_config AND f.liger=l.liger "
        "                AND f.finetuning_type='full' AND f.status='OOM')"
    ):
        conflicting.append(dict(row))
    print(f"  库中有 {len(conflicting)} 个「LoRA 成功但全参 OOM」的配置")
    check("存在这类冲突配置（否则本项无法验证）", len(conflicting) > 0,
          f"{len(conflicting)} 个")

    flagged = []
    for row in conflicting:
        entry = lc._catalog_entries().get(lc._normalise(row["model"]))
        if not entry:
            continue
        est = lora_model.estimate_lora(
            slug=entry["slug"], num_gpus=row["gpu_count"],
            micro_batch_size=row["bs"], seq_length=row["cutoff_len"],
            lora_rank=8, lora_alpha=16, lora_target="all",
            ds_config=row["ds_config"], liger=row["liger"],
        )
        if est.get("measured_oom"):
            flagged.append((entry["slug"], row["ds_config"], row["liger"],
                            row["cutoff_len"], row["bs"]))
    check("LoRA 估算未把全参 OOM 记到 LoRA 配置上",
          not flagged, str(flagged))

    conn.close()

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL MEASURED-MODE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
