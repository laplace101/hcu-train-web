"""Verify the exact request the browser sends when a preset model is picked.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_preset_solve.py http://127.0.0.1:7801
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7801"
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


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


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> int:
    print(f"=== 预设模型求解 @ {BASE} ===")

    boot = get("/api/bootstrap")
    presets = [m for m in boot["models"] if m.get("verified")] or boot["models"]
    slug = presets[0]["slug"]
    print(f"（使用预设模型 {slug}）")

    # This is exactly what the browser sends after picking a preset model:
    # model_slug is set, model_config_path is null.
    payload = {
        "mode": "sft_full",
        "hardware_id": "bw1000",
        "model_slug": slug,
        "model_config_path": None,
        "parallel": {"tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                     "num_gpus": 8, "micro_batch_size": 1,
                     "global_batch_size": 256, "seq_length": 4096},
    }

    status, body = post("/api/v1/training/search", payload)
    check("search 不再报 400/500", status == 200, str(status))
    check("ok=true", body.get("ok") is True, str(body.get("error"))[:120])
    if not body.get("ok"):
        return 1

    plans = body["plans"]
    for key in ("minimum", "stable", "performance"):
        plan = plans.get(key) or {}
        check(f"{key} 有卡数", plan.get("num_gpus"), str(plan.get("num_gpus")))
        check(f"{key} 有迭代时间", plan.get("iteration_ms"), str(plan.get("iteration_ms")))
        check(f"{key} 有吞吐", plan.get("tgs"), str(plan.get("tgs")))
        check(f"{key} 峰值显存不超 64GiB",
              (plan.get("peak_gib") or 0) <= 64, str(plan.get("peak_gib")))

    print("\n=== 同时验证 estimates 路径 ===")
    status, body = post("/api/v1/training/estimates", payload)
    check("estimates 200", status == 200 and body.get("ok"), str(status))
    if body.get("ok"):
        current = body["plans"]["current"]
        check("有 iteration_ms", current.get("iteration_ms"), str(current.get("iteration_ms")))
        check("有 tgs", current.get("tgs"), str(current.get("tgs")))
        check("有 peak_gib", current.get("peak_gib"), str(current.get("peak_gib")))

    print("\n=== 未找到的模型 slug 应给出清晰错误 ===")
    status, body = post("/api/v1/training/search",
                         dict(payload, model_slug="no-such-model"))
    check("返回 400", status == 400, str(status))
    check("错误提示可操作",
          "config" in body.get("error", {}).get("message", "").lower()
          or "建议" in str(body.get("error", {}).get("details", "")),
          body.get("error", {}).get("message", "")[:70])

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL PRESET-SOLVE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
