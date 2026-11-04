"""Verify the calibration tracks the growing measured database.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_calibration_refresh.py
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repositories import measured as measured_repo  # noqa: E402
from services import lora_calibration  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main() -> int:
    print("=== ① 校准随数据库自动重建 ===")
    # Build against the real database.
    first = lora_calibration.build_calibration(refresh=True)
    stamp = first.get("database_stamp")
    check("记录了数据库指纹", bool(stamp), str(stamp))
    check("记录了实测行数", (first.get("row_count") or 0) > 0,
          f"{first.get('row_count')} 行")
    check("产生了 step_time 拟合", len(first["step_time"]["fits"]) > 0,
          f"{len(first['step_time']['fits'])} 组")

    # A second call with no change must reuse the cache.
    second = lora_calibration.build_calibration()
    check("数据库未变时复用缓存", second.get("database_stamp") == stamp)

    print("\n=== ② 数据库变化后缓存失效 ===")
    # Work on a copy so the real database is never touched.
    source = measured_repo.database_path()
    tmp = Path(tempfile.mkdtemp()) / "bench_copy.db"
    shutil.copy(source, tmp)
    # The database runs in WAL mode; the sidecar carries the newest rows.
    wal = Path(f"{source}-wal")
    if wal.exists():
        shutil.copy(wal, Path(f"{tmp}-wal"))

    original_override = measured_repo.PATH_OVERRIDE
    try:
        measured_repo.PATH_OVERRIDE = tmp
        before = lora_calibration.build_calibration(refresh=True)
        before_rows = before["row_count"]

        # Add a row to the copy.
        conn = sqlite3.connect(tmp)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO train_bench (run_id, model, gpu_count, bs, grad_accum, "
            "cutoff_len, lora_rank, ds_config, liger, status, step_time_s, "
            "tokens_per_s, peak_mem_mib, ingested_at) "
            "VALUES (1, 'Qwen/Qwen3.5-2B', 8, 3, 8, 2048, 8, 'z3', 1, 'OK', "
            "18.0, 3600.0, 9000, '2026-11-04 12:00:00')"
        )
        conn.commit()
        conn.close()

        after = lora_calibration.build_calibration()
        check("数据库变化后缓存失效并重建",
              after.get("database_stamp") != before.get("database_stamp"),
              f"{before.get('database_stamp')} → {after.get('database_stamp')}")
        check("新数据被纳入校准", after["row_count"] > before_rows,
              f"{before_rows} → {after['row_count']}")
    finally:
        measured_repo.PATH_OVERRIDE = original_override
        shutil.rmtree(tmp.parent, ignore_errors=True)

    print("\n=== ③ 真实数据库未被写入 ===")
    conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    rows = conn.execute("SELECT COUNT(*) FROM train_bench").fetchone()[0]
    tampered = conn.execute(
        "SELECT COUNT(*) FROM train_bench WHERE note='TAMPERED'").fetchone()[0]
    conn.close()
    check("行数与读取时一致", rows > 0, f"{rows} 行")
    check("无篡改痕迹", tampered == 0, str(tampered))

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL CALIBRATION-REFRESH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
