"""Checks for the mode-agnostic measured comparison.

The comparison layer has to work for every training mode, and it has to be
honest about when two numbers cannot be compared. The measured database on
this machine holds LoRA runs only, so the full-parameter case is exercised
against a temporary copy of the schema — the real database is never
written to.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_measured_compare.py
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repositories import measured as measured_repo  # noqa: E402
from services import measured_compare as mc  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def db_checksum() -> str:
    path = measured_repo.database_path()
    if path is None:
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    before = db_checksum()

    print("=== ① 只读保证 ===")
    check("数据库可读", measured_repo.available())
    info = measured_repo.database_info()
    check("标记为只读", info.get("read_only") is True)

    print("\n=== ② 通用对照：与训练模式无关 ===")
    result = mc.compare(
        {"tgs": 2809.57, "peak_gib": 53.0},
        slug="qwen3-8b", gpu_count=8, seq_length=4096, mode="全参 SFT",
    )
    check("找到实测记录", result.get("matched") is True, str(result.get("matched")))
    check("给出估算吞吐", result.get("estimate_tgs") == 2809.57)
    check("给出实测吞吐", result.get("measured_tgs") is not None,
          str(result.get("measured_tgs")))
    check("给出偏差", result.get("tgs_deviation_percent") is not None,
          f"{result.get('tgs_deviation_percent')}%")
    check("标注匹配程度", result.get("match_quality") in {"exact", "nearest"},
          str(result.get("match_quality")))
    check("给出显存对照", result.get("measured_peak_gib") is not None,
          str(result.get("measured_peak_gib")))

    print("\n=== ③ 口径不同必须标出来 ===")
    # Every measured row on this machine is a LoRA run. Comparing it with a
    # full-parameter estimate has to say so.
    check("识别出实测是 LoRA", result.get("measured_is_lora") is True)
    check("标记为不可直接比较", result.get("comparable") is False)
    check("说明中提到口径不同",
          "不是同一口径" in (result.get("comparability_note") or ""),
          (result.get("comparability_note") or "")[:60])

    print("\n=== ④ 没有可比记录时如实说明 ===")
    empty = mc.compare({"tgs": 100.0}, slug="no-such-model-xyz", gpu_count=8)
    check("标记为无记录", empty.get("matched") is False)
    check("给出原因", bool(empty.get("reason")))
    check("给出建议", bool(empty.get("suggestion")))
    check("不编造数字", empty.get("measured_tgs") is None)

    print("\n=== ⑤ 全参实测数据接入后应当可比 ===")
    # The measured database holds LoRA runs only today. A full-parameter
    # run would be stored in the same table with finetuning_type='full'.
    # This exercises that path against a temporary copy of the schema, so
    # the real database stays untouched.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_db = Path(tmp) / "bench.db"
        source = measured_repo.database_path()
        if source is None:
            print("SKIP  没有实测数据库")
        else:
            # Copy the schema and the rows for one model, with the
            # finetuning type switched to 'full'.
            source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
            dest = sqlite3.connect(tmp_db)
            source_conn.backup(dest)
            source_conn.close()
            # `finetuning_type` lives on the base table; `v_train_bench`
            # is a view over it.
            dest.execute(
                "UPDATE train_bench SET finetuning_type='full', lora_rank=NULL")
            dest.commit()
            dest.close()

            original = measured_repo.DEFAULT_DB_PATHS
            try:
                measured_repo.DEFAULT_DB_PATHS = [tmp_db]
                full = mc.compare(
                    {"tgs": 2809.57, "peak_gib": 53.0},
                    slug="qwen3-8b", gpu_count=8, seq_length=4096,
                    mode="全参 SFT",
                )
            finally:
                measured_repo.DEFAULT_DB_PATHS = original

            check("全参记录被识别为非 LoRA",
                  full.get("measured_is_lora") is False,
                  str(full.get("measured_is_lora")))
            check("全参记录标记为可比", full.get("comparable") is True)
            check("全参记录仍给出偏差",
                  full.get("tgs_deviation_percent") is not None)
            check("全参记录不再提示口径不同",
                  "不是同一口径" not in (full.get("comparability_note") or ""))

    print("\n=== ⑥ 覆盖范围可查询 ===")
    env = mc.envelope()
    check("给出模型列表", isinstance(env.get("models"), list))
    check("给出模型数", env.get("model_count", 0) > 0, str(env.get("model_count")))

    print("\n=== ⑦ 数据库未被修改 ===")
    after = db_checksum()
    check("校验和不变", before == after, after[:16])

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL MEASURED-COMPARE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
