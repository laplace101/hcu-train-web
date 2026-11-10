"""Checks for the Qwen3.6 and Qwen3.8 series.

These models share the hybrid-attention architecture of Qwen3.5: the config
declares `model_type: qwen3_5` with a nested `text_config`, and the
kernel's own `qwen3_5` adapter handles them. The project's `qwen_hybrid`
family adapter covers them by capability, so they resolve to it.

Two things this test guards:

  * the catalogue entry carries the model's real dimensions — a VLM config
    keeps them inside `text_config`, and reading the top level would
    report zero layers and zero hidden size;
  * the estimate and the three-plan search both produce numbers, rather
    than failing on a config the kernel cannot represent.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_new_qwen_series.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402
from services import support_matrix  # noqa: E402

failures: list[str] = []

#: The series added for Qwen3.6 and Qwen3.8, with the shape each one has.
NEW_MODELS = [
    ("qwen3-6-27b", "Qwen3.6-27B", "Qwen3.6", "dense", 64, 5120, 24, 4, None),
    ("qwen3-6-35b-a3b", "Qwen3.6-35B-A3B", "Qwen3.6", "moe", 40, 2048, 16, 2, 256),
    ("qwen3-8-27b", "Qwen3.8-27B", "Qwen3.8", "dense", 64, 5120, 24, 4, None),
]


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def catalogue_entry(slug: str) -> dict | None:
    data = json.loads((PROJECT_ROOT / "catalog" / "catalog.json").read_text(encoding="utf-8"))
    for entry in data.get("models", []):
        if entry.get("slug") == slug:
            return entry
    return None


def main() -> int:
    client = app.test_client()

    print("=== ① 目录条目齐全 ===")
    for slug, name, family, kind, *_rest in NEW_MODELS:
        entry = catalogue_entry(slug)
        check(f"{slug} 在目录中", entry is not None)
        if entry is None:
            continue
        check(f"{slug} 显示名正确", entry.get("name") == name, str(entry.get("name")))
        check(f"{slug} 归到 {family} 族", entry.get("family") == family,
              str(entry.get("family")))
        check(f"{slug} 类型为 {kind}", entry.get("kind") == kind, str(entry.get("kind")))
        check(f"{slug} 已验证", entry.get("verified") is True)
        check(f"{slug} 走 qwen_hybrid 适配器", entry.get("adapter") == "qwen_hybrid",
              str(entry.get("adapter")))

    print("\n=== ② 目录里的维度是真的 ===")
    # A VLM config keeps the text model's dimensions inside `text_config`.
    # Reading the top level instead would report zero for all of them, and
    # the model would be published with no valid parallel options.
    for slug, _name, _family, _kind, layers, hidden, heads, kv, experts in NEW_MODELS:
        entry = catalogue_entry(slug)
        if entry is None:
            continue
        check(f"{slug} 层数 {layers}", entry.get("num_layers") == layers,
              str(entry.get("num_layers")))
        check(f"{slug} hidden {hidden}", entry.get("hidden_size") == hidden,
              str(entry.get("hidden_size")))
        check(f"{slug} 注意力头 {heads}", entry.get("num_attention_heads") == heads,
              str(entry.get("num_attention_heads")))
        check(f"{slug} KV 组 {kv}", entry.get("num_query_groups") == kv,
              str(entry.get("num_query_groups")))
        check(f"{slug} 专家数 {experts}", entry.get("num_experts") == experts,
              str(entry.get("num_experts")))
        check(f"{slug} 有合法 TP 选项", len(entry.get("valid_tp") or []) > 1,
              str(entry.get("valid_tp")))
        check(f"{slug} 有合法 PP 选项", len(entry.get("valid_pp") or []) > 1,
              str(entry.get("valid_pp")))
        check(f"{slug} 建议卡数为正", int(entry.get("suggested_num_gpus") or 0) > 0,
              str(entry.get("suggested_num_gpus")))

    print("\n=== ③ 支持矩阵判定为可估算 ===")
    for slug, *_rest in NEW_MODELS:
        support = support_matrix.check_reference(slug)
        check(f"{slug} 可估算", support.status == "estimable", support.status)
        check(f"{slug} 走 qwen_hybrid", support.adapter == "qwen_hybrid",
              str(support.adapter))

    print("\n=== ④ 全参估算给出数字 ===")
    for slug, _name, _family, _kind, *_rest in NEW_MODELS:
        config_path = str(PROJECT_ROOT / "catalog" / "models" / slug / "config.json")
        r = client.post("/api/v1/training/estimates", json={
            "mode": "sft_full",
            "hardware_id": "bw1000",
            "model_slug": slug,
            "model_config_path": config_path,
            "parallel": {"tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                         "num_gpus": 8, "micro_batch_size": 1,
                         "global_batch_size": 256, "seq_length": 4096},
        })
        body = r.get_json()
        check(f"{slug} 全参估算 200", r.status_code == 200 and body.get("ok") is True,
              str(body.get("error"))[:80])
        if not body.get("ok"):
            continue
        current = body["plans"]["current"]
        check(f"{slug} 给出迭代时间", (current.get("iteration_ms") or 0) > 0,
              str(current.get("iteration_ms")))
        check(f"{slug} 给出吞吐", (current.get("tgs") or 0) > 0, str(current.get("tgs")))
        check(f"{slug} 给出显存", (current.get("peak_gib") or 0) > 0,
              str(current.get("peak_gib")))

    print("\n=== ⑤ LoRA 三类方案都能求解 ===")
    for slug, _name, _family, _kind, *_rest in NEW_MODELS:
        config_path = str(PROJECT_ROOT / "catalog" / "models" / slug / "config.json")
        r = client.post("/api/v1/training/search", json={
            "mode": "sft_lora",
            "hardware_id": "bw1000",
            "model_slug": slug,
            "model_config_path": config_path,
            "lora": {"r": 8, "alpha": 16},
            "lora_ds_config": "z3",
            "lora_liger": 1,
            "parallel": {"tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                         "num_gpus": 8, "micro_batch_size": 1,
                         "global_batch_size": 256, "seq_length": 2048},
        })
        body = r.get_json()
        check(f"{slug} LoRA 求解 200", r.status_code == 200 and body.get("ok") is True,
              str(body.get("error"))[:80])
        if not body.get("ok"):
            continue
        plans = body["plans"]
        for name in ("minimum", "stable", "performance"):
            plan = plans.get(name) or {}
            check(f"{slug} {name} 方案有卡数", (plan.get("num_gpus") or 0) > 0)
            check(f"{slug} {name} 方案有吞吐", (plan.get("tokens_per_s") or 0) > 0,
                  str(plan.get("tokens_per_s")))
            check(f"{slug} {name} 方案有显存", (plan.get("peak_gib") or 0) > 0,
                  str(plan.get("peak_gib")))
            # A plan the campaign measured as out of memory must not be
            # offered as runnable.
            check(f"{slug} {name} 方案未被判为实测 OOM",
                  plan.get("measured_oom") is not True, str(plan.get("measured_oom")))

    print("\n=== ⑥ 实测对照的口径正确 ===")
    # A LoRA estimate is compared against LoRA measurements. The 3.8 and
    # 3.6 campaigns measured LoRA runs, so the comparison is like for like.
    for slug, _name, _family, _kind, *_rest in NEW_MODELS:
        config_path = str(PROJECT_ROOT / "catalog" / "models" / slug / "config.json")
        r = client.post("/api/v1/training/search", json={
            "mode": "sft_lora",
            "hardware_id": "bw1000",
            "model_slug": slug,
            "model_config_path": config_path,
            "lora": {"r": 8, "alpha": 16},
            "lora_ds_config": "z3",
            "lora_liger": 1,
            "parallel": {"tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                         "num_gpus": 8, "micro_batch_size": 1,
                         "global_batch_size": 256, "seq_length": 2048},
        })
        body = r.get_json()
        measured = body.get("measured") or {}
        if not measured.get("matched"):
            # A model with no measured run yet is not an error; the
            # comparison simply has nothing to say.
            print(f"  SKIP  {slug} 实测库中暂无记录")
            continue
        check(f"{slug} 匹配到 LoRA 实测", measured.get("measured_is_lora") is True,
              str(measured.get("measured_is_lora")))
        check(f"{slug} 同口径可比", measured.get("comparable") is True,
              str(measured.get("comparable")))

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL NEW-SERIES CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
