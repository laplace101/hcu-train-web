"""End-to-end checks for the layered /api/v1 API (plan C/D coverage).

Runs against the Flask test client so no server needs to be running.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_api_v1.py
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


def post(path: str, payload: dict):
    with app.test_client() as client:
        response = client.post(path, json=payload)
        return response.status_code, response.get_json()


def get(path: str):
    with app.test_client() as client:
        response = client.get(path)
        return response.status_code, response.get_json()


def main() -> int:
    print("=== ① 元数据 ===")
    status, body = get("/api/v1/health")
    check("health 200", status == 200, str(status))
    check("模拟器可用", body["simulator"]["available"] is True, body["simulator"]["version"])
    check("六种训练类型", len(body["modes"]) == 6, str(len(body["modes"])))
    modelled = [m["mode"] for m in body["modes"] if m["modelled"]]
    check("已建模：预训练/全参/LoRA", set(modelled) == {"pretrain", "sft_full", "sft_lora"},
          str(modelled))
    unmodelled = {m["mode"] for m in body["modes"] if not m["modelled"]}
    check("未建模：DPO/GRPO/PPO", unmodelled == {"dpo", "grpo", "ppo"},
          str(sorted(unmodelled)))
    check("硬件目录含 BW1000", any(h["id"] == "bw1000" and h["supported"] for h in body["hardware"]))
    check("未支持卡带原因", all(
        h["unsupported_reason"] for h in body["hardware"] if not h["supported"]
    ))

    print("\n=== ② 参数校验（C05/C06）===")
    base = {
        "mode": "sft_full",
        "hardware_id": "bw1000",
        "model_config_path": str(PROJECT_ROOT / "catalog" / "models" / "qwen3-8b" / "config.json"),
        "parallel": {
            "tp_size": 2, "cp_size": 1, "pp_size": 1, "ep_size": 1,
            "num_gpus": 8, "micro_batch_size": 1, "global_batch_size": 256,
            "seq_length": 4096,
        },
    }
    status, body = post("/api/v1/training/validate", base)
    check("合法参数通过", status == 200 and body["ok"] is True, str(body.get("issues")))

    bad_tp = dict(base, parallel=dict(base["parallel"], tp_size=5))
    status, body = post("/api/v1/training/validate", bad_tp)
    issues = body.get("issues", [])
    check("TP 不整除 KV 头被拦截",
          any(i["field"] == "tp_size" and i["severity"] == "error" for i in issues),
          "; ".join(i["message"] for i in issues)[:110])
    check("拦截项带修改建议",
          any(i.get("suggestion") for i in issues))

    # base has TP=2 on 8 GPUs -> DP=4, so a GBS that is not a multiple of 4
    # is the illegal case (100 would be legal here: 100 % 4 == 0).
    bad_gbs = dict(base, parallel=dict(base["parallel"], global_batch_size=101))
    status, body = post("/api/v1/training/validate", bad_gbs)
    issues = body.get("issues", [])
    check("GBS 不整除被拦截",
          any(i["field"] == "global_batch_size" and i["severity"] == "error" for i in issues),
          "; ".join(i["message"] for i in issues if i["severity"] == "error")[:110])
    check("GBS 拦截项给出可用值",
          any("最近可用值" in i.get("suggestion", "") for i in issues),
          next((i["suggestion"] for i in issues if i["field"] == "global_batch_size"), "")[:80])

    bad_gpus = dict(base, parallel=dict(base["parallel"], num_gpus=6))
    status, body = post("/api/v1/training/validate", bad_gpus)
    issues = body.get("issues", [])
    check("非整节点卡数被警告",
          any(i["field"] == "num_gpus" and i["severity"] == "warning" for i in issues),
          "; ".join(i["message"] for i in issues)[:100])
    check("非法组合导致 ok=False", body.get("ok") is False)

    # 6 GPUs is not a multiple of TP=2 -> DP=3 -> GBS 256 not divisible by 3.
    check("DP 不整除时 GBS 被拦截",
          any(i["field"] == "global_batch_size" and i["severity"] == "error" for i in issues),
          "; ".join(i["message"] for i in issues if i["severity"] == "error")[:100])

    print("\n=== ③ 未建模模式明确拒绝（规划书 §1.3）===")
    # LoRA has its own resource model now, so it is estimated rather than
    # refused; the RL modes still have no kernel.
    for mode in ("dpo", "grpo", "ppo"):
        status, body = post("/api/v1/training/search", dict(base, mode=mode))
        err = (body or {}).get("error") or {}
        check(f"{mode} 返回 422 且说明原因",
              status == 422 and err.get("code") == "mode_not_modelled"
              and len(err.get("message", "")) > 20,
              err.get("message", "")[:70])

    status, body = post("/api/v1/training/search", dict(base, mode="sft_lora"))
    check("LoRA 不再被拒绝", status == 200 and (body or {}).get("ok") is True,
          f"HTTP {status}")

    print("\n=== ④ 不支持的硬件拒绝 ===")
    status, body = post("/api/v1/training/search", dict(base, hardware_id="bw1100"))
    check("BW1100 被拒并说明原因",
          status == 400 and "带宽表" in body["error"]["message"],
          body.get("error", {}).get("message", "")[:90])

    print("\n=== ⑤ 当前配置仿真（estimates）===")
    status, body = post("/api/v1/training/estimates", base)
    check("仿真 200", status == 200 and body.get("ok") is True, str(status))
    if body.get("ok"):
        current = body["plans"]["current"]
        check("有 iteration 时间", current.get("iteration_ms", 0) > 0, f"{current.get('iteration_ms')}")
        check("有 MFU", 0 < (current.get("mfu") or 0) < 1, f"{(current.get('mfu') or 0) * 100:.2f}%")
        check("有 TGS", (current.get("tgs") or 0) > 0, f"{current.get('tgs')}")
        check("有峰值显存", (current.get("peak_gib") or 0) > 0, f"{current.get('peak_gib')} GiB")
        check("记录模拟器版本", bool(body.get("simulator_version")), body.get("simulator_version", ""))

    print("\n=== ⑥ 三类方案搜索（search）===")
    status, body = post("/api/v1/training/search", base)
    check("搜索 200", status == 200 and body.get("ok") is True, str(status))
    if body.get("ok"):
        plans = body["plans"]
        check("有最低可训练方案", bool(plans.get("minimum")), str(plans.get("minimum", {}).get("num_gpus")))
        check("有推荐稳定方案", bool(plans.get("stable")), str(plans.get("stable", {}).get("num_gpus")))
        check("有推荐性能方案", bool(plans.get("performance")), str(plans.get("performance", {}).get("num_gpus")))
        perf = plans.get("performance") or {}
        minimum = plans.get("minimum") or {}
        check("性能方案 TGS 不低于最低方案",
              (perf.get("tgs") or 0) >= (minimum.get("tgs") or 0),
              f"{minimum.get('tgs')} → {perf.get('tgs')}")
        check("峰值显存不超单卡 64 GiB",
              all((p.get("peak_gib") or 0) <= 64 for p in (minimum, perf)),
              f"min={minimum.get('peak_gib')} perf={perf.get('peak_gib')}")
        meta = plans.get("_meta") or {}
        check("稳定方案标注了标签", bool(meta.get("stable_label")), meta.get("stable_label", ""))

    print("\n=== ⑦ 模型 config 解析 ===")
    status, body = post("/api/v1/model-configs/resolve", {
        "source": "upload",
        "config_text": '{"model_type":"qwen3","num_hidden_layers":36,"hidden_size":4096,'
                       '"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936}',
    })
    check("上传 config 可解析", status == 200 and body.get("ok") is True)
    check("产出 config 哈希", len(body["provenance"]["config_hash"]) == 64)
    check("识别出层数", body["facts"]["num_hidden_layers"] == 36)

    status, body = post("/api/v1/model-configs/resolve", {"source": "upload", "config_text": "{bad json"})
    check("非法 JSON 返回明确错误",
          status == 400 and body["error"]["code"] == "model_source_failed",
          body.get("error", {}).get("message", "")[:70])

    status, body = post("/api/v1/model-configs/resolve", {
        "source": "modelscope", "reference": "https://evil.example.com/models/x/y",
    })
    check("非白名单域名被拒",
          status == 400 and "不支持的域名" in body["error"]["message"],
          body.get("error", {}).get("message", "")[:70])

    print("\n=== ⑧ 缺失模型时报错 ===")
    no_model = {k: v for k, v in base.items() if k != "model_config_path"}
    status, body = post("/api/v1/training/estimates", no_model)
    check("缺 config 报错",
          status == 400 and "config" in body["error"]["message"].lower(),
          body.get("error", {}).get("message", "")[:80])

    print("\n=== ⑨ 适配层完整性（回归）===")
    # The isolated runner is a module-level constant. It was once lost when a
    # patch replaced a slice that happened to span it, which broke every
    # estimate at runtime. Check the wiring explicitly.
    from estimators import adapter

    check("adapter 有 _RUNNER 常量", hasattr(adapter, "_RUNNER"),
          "缺失会导致所有估算 500")
    check("_RUNNER 是字符串", isinstance(getattr(adapter, "_RUNNER", None), str))
    runner = getattr(adapter, "_RUNNER", "")
    check("runner 引用 engine 模块", "from engine import engine" in runner)
    check("runner 使用结果标记", "RESULT_MARKER" in runner)
    check("runner 处理两种模式",
          'mode == "solve"' in runner and "engine.solve(" in runner
          and "engine.simulate(" in runner,
          "runner 必须区分 solve 与 simulate")
    check("summarise_simulation 存在", hasattr(adapter, "summarise_simulation"))

    print("\n=== ⑩ 内核共享标注（如实告知）===")
    _status, _modes_body = get("/api/v1/training/modes")
    modes = {m["mode"]: m for m in _modes_body["modes"]}
    check("预训练有 kernel_note", bool(modes["pretrain"].get("kernel_note")))
    check("全参 SFT 有 kernel_note", bool(modes["sft_full"].get("kernel_note")))
    check("两模式共用同一段说明",
          modes["pretrain"]["kernel_note"] == modes["sft_full"]["kernel_note"])
    check("未建模模式不显示该标注",
          not modes["grpo"].get("kernel_note")
          and not modes["ppo"].get("kernel_note"))
    check("标注说明了数值一致",
          "一致" in modes["pretrain"]["kernel_note"],
          modes["pretrain"]["kernel_note"][:40])
    check("标注说明了实际差别",
          "训练步数" in modes["pretrain"]["kernel_note"])

    print("\n=== ⑪ 旧接口仍可用（回归基线依赖）===")
    status, body = get("/api/health")
    check("旧 /api/health 200", status == 200 and body.get("ok") is True)
    status, body = get("/api/bootstrap")
    check("旧 /api/bootstrap 200", status == 200 and body.get("ok") is True)
    check("bootstrap 含 35 个预设模型", len(body.get("models", [])) == 35, str(len(body.get("models", []))))

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL API v1 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
