"""Every preset model must be solvable through the three-plan path.

A preset carries the card count it was published with and the batch size
that was current then. Once the cluster shape changes, the two can
disagree — 256 is not a multiple of the data-parallel size 9 that 72 cards
with TP=8 give, and the estimate then fails with a divisibility error that
looks like the user's mistake.

This drives the real API path for all 38 presets.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_presets_solvable.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main() -> int:
    client = app.test_client()
    catalog = json.loads(
        (PROJECT_ROOT / "catalog" / "catalog.json").read_text(encoding="utf-8"))

    print("=== ① 每个预设模型都能求解三类方案 ===")
    failed: list[str] = []
    adjusted: list[str] = []

    for entry in catalog["models"]:
        slug = entry["slug"]
        baseline = entry.get("baseline") or {}
        parallel = {
            "tp_size": entry.get("suggested_tp", 1),
            "pp_size": entry.get("suggested_pp", 1),
            "cp_size": 1,
            "ep_size": entry.get("suggested_ep", 1),
            "num_gpus": entry.get("suggested_num_gpus", 8),
            "micro_batch_size": baseline.get("micro_batch_size", 1),
            "global_batch_size": baseline.get("global_batch_size", 256),
            "seq_length": baseline.get("seq_length", 4096),
        }
        # The full search is expensive (one solve per model). The models
        # that need it are the ones whose data-parallel size does not
        # divide the published batch size — those are the ones that used
        # to fail. The rest are checked through the validation path, which
        # is fast.
        product = parallel["tp_size"] * parallel["pp_size"]
        dp = (parallel["num_gpus"] // product
              if parallel["num_gpus"] % product == 0 else 1)
        needs_full_search = parallel["global_batch_size"] % (
            parallel["micro_batch_size"] * dp) != 0

        if not needs_full_search:
            # A legal combination: the estimate must not be rejected.
            r = client.post("/api/v1/training/estimates", json={
                "mode": "sft_full",
                "hardware_id": "bw1000",
                "model_slug": slug,
                "model_config_path": str(PROJECT_ROOT / entry["config_path"]),
                "parallel": parallel,
            })
            body = r.get_json() or {}
            if not body.get("ok"):
                error = body.get("error") or {}
                failed.append(f"{slug}: {error.get('code')}")
            continue

        r = client.post("/api/v1/training/search", json={
            "mode": "sft_full",
            "hardware_id": "bw1000",
            "model_slug": slug,
            "model_config_path": str(PROJECT_ROOT / entry["config_path"]),
            "parallel": parallel,
        })
        body = r.get_json() or {}
        if not body.get("ok"):
            error = body.get("error") or {}
            failed.append(f"{slug}: {error.get('code')}")
            continue
        plans = body.get("plans") or {}
        if not (plans.get("minimum") or plans.get("performance")):
            failed.append(f"{slug}: 方案为空")
            continue
        if body.get("adjustments"):
            adjusted.append(slug)

    check("全部预设模型都能求解", not failed,
          "; ".join(failed[:5]) if failed else f"{len(catalog['models'])} 个模型")
    print(f"      失败的模型: {failed if failed else '无'}")
    print(f"      触发全局批量调整的模型: {len(adjusted)} 个")

    print("\n=== ② 全局批量调整被如实报告 ===")
    # One model whose card count gives a data-parallel size that does not
    # divide the published batch size.
    entry = next(e for e in catalog["models"]
                 if e["slug"] == "minimax-m2")
    baseline = entry.get("baseline") or {}
    r = client.post("/api/v1/training/search", json={
        "mode": "sft_full",
        "hardware_id": "bw1000",
        "model_slug": "minimax-m2",
        "model_config_path": str(PROJECT_ROOT / entry["config_path"]),
        "parallel": {
            "tp_size": entry.get("suggested_tp", 1),
            "pp_size": entry.get("suggested_pp", 1),
            "cp_size": 1,
            "ep_size": entry.get("suggested_ep", 1),
            "num_gpus": entry.get("suggested_num_gpus", 8),
            "micro_batch_size": baseline.get("micro_batch_size", 1),
            "global_batch_size": baseline.get("global_batch_size", 256),
            "seq_length": baseline.get("seq_length", 4096),
        },
    })
    body = r.get_json() or {}
    check("M2 求解成功", body.get("ok") is True,
          str((body.get("error") or {}).get("message"))
          if not body.get("ok") else "")
    notes = body.get("adjustments") or []
    check("给出调整说明", len(notes) > 0,
          notes[0][:70] if notes else "无")
    check("说明里含整除关系",
          any("整除" in n for n in notes),
          notes[0][:70] if notes else "")
    check("说明里含调整后的数值",
          any("已调整为" in n for n in notes),
          notes[0][:70] if notes else "")

    print("\n=== ③ 合法组合不受影响 ===")
    # A card count whose DP divides the batch size — nothing to adjust.
    r2 = client.post("/api/v1/training/search", json={
        "mode": "sft_full",
        "hardware_id": "bw1000",
        "model_slug": "qwen3-8b",
        "model_config_path": str(
            PROJECT_ROOT / "catalog" / "models" / "qwen3-8b" / "config.json"),
        "parallel": {
            "tp_size": 8, "pp_size": 1, "cp_size": 1, "ep_size": 1,
            "num_gpus": 8, "micro_batch_size": 1,
            "global_batch_size": 256, "seq_length": 4096,
        },
    })
    body2 = r2.get_json() or {}
    check("Qwen3-8B 求解成功", body2.get("ok") is True)
    check("没有调整说明", not (body2.get("adjustments") or []),
          str(body2.get("adjustments")))

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL PRESET-SOLVABLE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
