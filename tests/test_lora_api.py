"""API-level checks for the LoRA module.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_lora_api.py
"""

from __future__ import annotations

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

    print("=== ① 模式目录 ===")
    response = client.get("/api/v1/training/modes")
    status, body = response.status_code, response.get_json()
    modes = {m["mode"]: m for m in body["modes"]}
    check("LoRA 已标记为可建模", modes["sft_lora"]["modelled"] is True)
    check("LoRA 有内核说明", bool(modes["sft_lora"].get("kernel_note")),
          modes["sft_lora"].get("kernel_note", "")[:50])
    check("LoRA 不再是未建模原因", not modes["sft_lora"].get("unmodelled_reason"))
    check("GRPO 仍未建模", modes["grpo"]["modelled"] is False)

    print("\n=== ② LoRA 搜索接口 ===")
    response = client.post("/api/v1/training/search", json={
        "mode": "sft_lora",
        "hardware_id": "bw1000",
        "model_slug": "qwen3-5-2b",
        "model_config_path": str(
            PROJECT_ROOT / "catalog" / "models" / "qwen3-5-2b" / "config.json"),
        "parallel": {"tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                     "num_gpus": 8, "micro_batch_size": 4,
                     "global_batch_size": 256, "seq_length": 2048},
        "lora": {"r": 8, "alpha": 16},
    })
    status, body = response.status_code, response.get_json()
    check("search 200", status == 200, str(status))
    check("ok=true", body.get("ok") is True, str(body.get("error"))[:150])
    if body.get("ok"):
        plans = body["plans"]
        for key in ("minimum", "stable", "performance"):
            plan = plans.get(key) or {}
            check(f"{key} 有卡数", plan.get("num_gpus"), str(plan.get("num_gpus")))
            check(f"{key} 有单步时间", plan.get("step_time_s"), str(plan.get("step_time_s")))
            check(f"{key} 有吞吐", plan.get("tokens_per_s"), str(plan.get("tokens_per_s")))
            check(f"{key} 有显存", plan.get("peak_gib"), str(plan.get("peak_gib")))
        check("给出校准摘要", bool(body.get("calibration")),
              f"实测行数 {body.get('calibration', {}).get('measured_rows')}")
        check("标注为 LoRA 校准", "lora" in (body.get("simulator_version") or ""),
              body.get("simulator_version", ""))

    print("\n=== ③ estimates 接口 ===")
    response = client.post("/api/v1/training/estimates", json={
        "mode": "sft_lora",
        "hardware_id": "bw1000",
        "model_slug": "qwen3-5-2b",
        "parallel": {"tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                     "num_gpus": 8, "micro_batch_size": 4,
                     "global_batch_size": 256, "seq_length": 2048},
        "lora": {"r": 8, "alpha": 16},
    })
    status, body = response.status_code, response.get_json()
    check("estimates 200", status == 200 and body.get("ok"), str(status))
    if body.get("ok"):
        current = body["plans"]["current"]
        check("current 有单步时间", current.get("step_time_s"), str(current.get("step_time_s")))
        check("current 有吞吐", current.get("tokens_per_s"), str(current.get("tokens_per_s")))
        check("current 有显存", current.get("peak_gib"), str(current.get("peak_gib")))
        check("current 标注显存来源", bool(current.get("memory_source")),
              current.get("memory_source", ""))

    print("\n=== ④ 附带实测数据 ===")
    response = client.post("/api/v1/training/search", json={
        "mode": "sft_lora",
        "hardware_id": "bw1000",
        "model_slug": "qwen3-5-2b",
        "parallel": {"num_gpus": 8, "micro_batch_size": 4, "seq_length": 2048},
        "lora": {"r": 8, "alpha": 16},
    })
    status, body = response.status_code, response.get_json()
    m = body.get("measured") or {}
    check("带 measured 段", m.get("available") is True)
    check("匹配到实测模型", m.get("matched") is True, str(m.get("slug")))

    print("\n=== ⑤ 未覆盖模型的处理 ===")
    response = client.post("/api/v1/training/search", json={
        "mode": "sft_lora",
        "hardware_id": "bw1000",
        "model_slug": "qwen3-14b",
        "parallel": {"num_gpus": 8, "micro_batch_size": 1, "seq_length": 4096},
        "lora": {"r": 8, "alpha": 16},
    })
    status, body = response.status_code, response.get_json()
    check("未实测模型仍给结果或明确拒绝", status in (200, 400),
          f"HTTP {status}")
    if status == 200 and body.get("ok"):
        perf = body["plans"].get("performance") or {}
        conf = perf.get("throughput_confidence") or {}
        check("跨模型外推有明确标注",
              conf.get("confidence") in ("cross_model_extrapolation", "measured_fit"),
              str(conf.get("confidence")))
        check("外推标注了规模差距", conf.get("params_gap_b") is not None,
              str(conf.get("params_gap_b")))
    else:
        check("拒绝时给出原因", bool(body.get("error", {}).get("message")),
              body.get("error", {}).get("message", "")[:60])

    print("\n=== ⑥ 缺失模型标识时报错 ===")
    response = client.post("/api/v1/training/search", json={
        "mode": "sft_lora",
        "hardware_id": "bw1000",
        "parallel": {"num_gpus": 8, "micro_batch_size": 1, "seq_length": 4096},
    })
    status, body = response.status_code, response.get_json()
    check("缺模型时报 400", status == 400, str(status))
    details = body.get("error", {}).get("details") or []
    check("给出可操作建议",
          bool(details) and bool(details[0].get("suggestion")),
          str(details[0].get("suggestion", ""))[:70] if details else "")

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL LORA-API CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
