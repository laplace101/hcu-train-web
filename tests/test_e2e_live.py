"""End-to-end check against a running server (not the test client).

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_e2e_live.py http://127.0.0.1:7803
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7803"

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def get(path: str):
    try:
        with urllib.request.urlopen(BASE + path, timeout=90) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def post(path: str, payload: dict):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=280) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def main() -> int:
    print(f"=== 实时端到端 @ {BASE} ===")

    status, health = get("/api/v1/health")
    check("health 200", status == 200)
    check("模拟器可用", health["simulator"]["available"] is True, health["simulator"]["version"])
    check("六种训练类型", len(health["modes"]) == 6)

    print("\n=== 估算（真实 simulator 调用）===")
    status, body = post("/api/v1/training/estimates", {
        "mode": "sft_full",
        "hardware_id": "bw1000",
        "model_config_path": "/root/hygon/train_web/hcu-train-web/catalog/models/qwen3-8b/config.json",
        "parallel": {"tp_size": 2, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                     "num_gpus": 8, "micro_batch_size": 1,
                     "global_batch_size": 256, "seq_length": 4096},
    })
    check("estimates 200", status == 200 and body.get("ok"), str(status))
    if body.get("ok"):
        c = body["plans"]["current"]
        check("iteration_ms 与基线一致",
              abs(c["iteration_ms"] - 46651.92) / 46651.92 < 0.005,
              f"{c['iteration_ms']:.2f} (基线 46651.92)")
        check("tgs 与基线一致",
              abs(c["tgs"] - 2809.57) / 2809.57 < 0.01, f"{c['tgs']:.2f}")
        check("峰值显存与基线一致",
              abs(c["peak_gib"] - 53.0) / 53.0 < 0.01, f"{c['peak_gib']}")

    print("\n=== 三类方案 ===")
    status, body = post("/api/v1/training/search", {
        "mode": "sft_full",
        "hardware_id": "bw1000",
        "model_config_path": "/root/hygon/train_web/hcu-train-web/catalog/models/qwen3-8b/config.json",
        "parallel": {"tp_size": 2, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                     "num_gpus": 8, "micro_batch_size": 1,
                     "global_batch_size": 256, "seq_length": 4096},
        "total_tokens": 1000000000,
        "epochs": 1,
    })
    check("search 200", status == 200 and body.get("ok"), str(status))
    plans = body.get("plans", {}) if body.get("ok") else {}
    check("三类方案齐全", all(k in plans for k in ("minimum", "stable", "performance")),
          ", ".join(plans.keys()))
    gpu_counts = {k: plans.get(k, {}).get("num_gpus") for k in ("minimum", "stable", "performance")}
    check("方案有卡数", all(gpu_counts.values()), str(gpu_counts))

    print("\n=== 未建模模式拒绝 ===")
    status, body = post("/api/v1/training/search", {
        "mode": "grpo",
        "hardware_id": "bw1000",
        "model_config_path": "/root/hygon/train_web/hcu-train-web/catalog/models/qwen3-8b/config.json",
        "parallel": {"tp_size": 2, "num_gpus": 8},
    })
    check("GRPO 返回 422", status == 422, str(status))
    check("说明未建模原因",
          "error" in body and len(body["error"].get("message", "")) > 15,
          body.get("error", {}).get("message", "")[:60])

    print("\n=== 启动命令渲染 ===")
    plan = plans.get("performance") or {}
    status, body = post("/api/v1/launch", {
        "template_id": "megatron-sft-full",
        "plan": plan,
        "model_path": "/models/Qwen3-8B",
        "output_dir": "/workspace/output",
        "work_dir": "/workspace",
        "image": "harbor.sourcefind.cn:5443/hcu/admin/base/custom:dtk26.04-py3.10",
    })
    check("launch 200", status == 200 and body.get("ok"), str(status))
    if body.get("ok"):
        check("YAML 含并行参数", "tensor_model_parallel_size" in body["yaml"])
        check("Shell 可执行", body["shell"].startswith("#!/usr/bin/env bash"))
        check("Docker 含镜像", "harbor.sourcefind.cn" in body["docker"])
        check("rendezvous 正确", body["rendezvous"].startswith("torchrun"))

    print("\n=== 镜像目录 ===")
    status, body = get("/api/v1/images")
    check("images 200", status == 200 and body["ok"])
    check("有训练镜像", body["count"] > 0, f"{body['count']} 个仓库")
    check("tag 可拉取", all(t["pull_url"] for i in body["images"] for t in i["tags"]))

    print("\n=== 页面加载 ===")
    with urllib.request.urlopen(BASE + "/", timeout=30) as r:
        html = r.read().decode()
    check("页面 200", r.status == 200)
    for marker in ['id="task-grid"', 'data-tab="plans"', 'data-tab="validate"',
                   'data-tab="launch"', 'id="data-grid"', 'id="img-framework"',
                   'id="tpl-select"', 'data-model-tab="modelscope"']:
        check(f"页面含 {marker}", marker in html)

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL LIVE E2E CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
