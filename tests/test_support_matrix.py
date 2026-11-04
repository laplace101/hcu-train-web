"""Checks for the model support matrix.

The matrix must reflect what the kernel can actually do, not a stored
opinion. These checks compare it against the simulator.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_support_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402
from services import compatibility  # noqa: E402
from services import support_matrix  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main() -> int:
    client = app.test_client()

    print("=== ① 矩阵端点 ===")
    r = client.get("/api/v1/model-support")
    body = r.get_json()
    check("HTTP 200", r.status_code == 200)
    check("ok 为真", body.get("ok") is True)
    totals = body.get("totals") or {}
    check("有统计", totals.get("total", 0) > 0, str(totals))
    check("列出适配器", len(body.get("adapters") or []) >= 5)
    check("按系列分组", len(body.get("families") or {}) >= 3)

    # Every model must land in a family bucket, and the bucket totals must
    # add up — otherwise the filter would hide models.
    families = body.get("families") or {}
    models = body.get("models") or []
    model_families = {row.get("family") for row in models}
    check("每个模型都有系列", all(f for f in model_families),
          str(sorted(f for f in model_families if not f)))
    check("系列覆盖所有模型",
          model_families <= set(families),
          str(sorted(model_families - set(families))))
    counted = sum(
        bucket.get("estimable", 0) + bucket.get("approximate", 0)
        + bucket.get("not_estimable", 0)
        for bucket in families.values()
    )
    check("系列统计与总数一致", counted == len(models),
          f"{counted} vs {len(models)}")

    print("\n=== ② 每个预设模型的判定与内核一致 ===")
    # The matrix's verdict for a preset model must match the compatibility
    # check, and the adapter must match what the kernel resolves.
    mismatched = []
    for row in body.get("models", []):
        slug = row["reference"]
        direct = support_matrix.check_reference(slug)
        if direct.adapter != row["adapter"] or direct.status != row["status"]:
            mismatched.append(slug)
    check("矩阵与直接判定一致", not mismatched, ", ".join(mismatched[:3]))

    print("\n=== ③ 已知模型的判定 ===")
    expectations = [
        ("qwen3-8b", "estimable", "qwen3"),
        ("qwen3-5-2b", "estimable", "qwen3_5"),
        ("qwen3-vl-8b-instruct", "estimable", "qwen3_vl"),
        ("deepseek-v3", "estimable", "deepseek_v3"),
        ("glm-5", "estimable", "glm5"),
        ("qwen2-5-7b", "approximate", "generic_gpt"),
    ]
    for slug, status, adapter in expectations:
        support = support_matrix.check_reference(slug)
        check(f"{slug} → {status}", support.status == status,
              f"实际 {support.status}（{support.adapter}）")
        check(f"{slug} → 适配器 {adapter}", support.adapter == adapter,
              f"实际 {support.adapter}")

    print("\n=== ④ 不可估算的模型给出原因 ===")
    # DeepSeek-V3.2 has DSA index attention; only the glm5 adapter applies it.
    v32 = support_matrix.check_reference("deepseek-v3-2-exp")
    check("V3.2 判为不可估算", v32.status == "not_estimable", v32.status)
    check("列出原因", len(v32.reasons) > 0, v32.reasons[0][:60] if v32.reasons else "")
    check("给出建议", len(v32.suggestions) > 0)

    print("\n=== ⑤ 单模型查询端点 ===")
    r = client.post("/api/v1/model-support/check", json={"reference": "qwen3-8b"})
    check("HTTP 200", r.status_code == 200)
    check("状态可估", r.get_json().get("status") == "estimable")

    r = client.post("/api/v1/model-support/check", json={"reference": "no-such-model"})
    check("未知模型仍是 200", r.status_code == 200)
    check("未知模型判为不可估", r.get_json().get("status") == "not_estimable")
    check("给出原因", len(r.get_json().get("reasons") or []) > 0)

    r = client.post("/api/v1/model-support/check", json={})
    check("缺参数返回 400", r.status_code == 400)

    print("\n=== ⑥ 适配器能力端点 ===")
    r = client.get("/api/v1/model-support/adapters")
    adapters = (r.get_json() or {}).get("adapters") or {}
    check("HTTP 200", r.status_code == 200)
    check("列出 6 个适配器", len(adapters) == 6, str(len(adapters)))
    check("glm5 支持 DSA", adapters["glm5"]["supports_dsa"] is True)
    check("qwen3_vl 支持视觉", adapters["qwen3_vl"]["supports_vision"] is True)
    check("generic 是兜底", adapters["generic_gpt"]["is_fallback"] is True)
    check("每个适配器都说明读哪些字段",
          all(adapters[a]["reads"] for a in adapters))

    print("\n=== ⑨ 上游对照 ===")
    # The upstream simulator ships the same six adapters, so the limit is
    # the kernel's coverage rather than this tool's.
    r = client.get("/api/v1/model-support/upstream")
    upstream = r.get_json()
    check("上游对照 HTTP 200", r.status_code == 200)
    check("上游同样是 6 个适配器", upstream.get("adapter_count") == 6,
          str(upstream.get("adapter_count")))
    check("列出适配器", len(upstream.get("adapters") or []) == 6)
    check("记录兜底适配器", upstream.get("fallback_adapter") == "generic_gpt")
    check("说明上游降级行为",
          bool(upstream.get("fallback_behaviour")))
    check("说明本项目的门禁行为",
          bool(upstream.get("gate_behaviour")))

    # The matrix payload carries the same comparison.
    check("矩阵里带上游对照",
          bool((body.get("upstream") or {}).get("fallback_adapter")))

    print("\n=== ⑦ 阻断与提示分离 ===")
    # A model the kernel can handle but whose adapter ignores some fields
    # must still be estimable, with a warning rather than a refusal.
    config = json.loads(
        (PROJECT_ROOT / "catalog" / "models" / "deepseek-v3" / "config.json")
        .read_text(encoding="utf-8")
    )
    verdict = compatibility.check_compatibility(config)
    check("DeepSeek-V3 可估算", verdict["supported"] is True,
          str(verdict["reasons"])[:70])
    check("DeepSeek-V3 无阻断原因", verdict["reasons"] == [],
          str(verdict["reasons"])[:70])

    print("\n=== ⑧ 次要特性只提示不拦截 ===")
    # Sliding-window attention is not modelled, but it does not change the
    # attention layout, so the model stays estimable with a warning.
    sliding = {"model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"],
               "num_hidden_layers": 48, "hidden_size": 3840,
               "num_attention_heads": 16, "num_key_value_heads": 8,
               "intermediate_size": 9216, "vocab_size": 262144,
               "sliding_window": 1024}
    verdict_sliding = compatibility.check_compatibility(sliding)
    check("滑窗模型仍可估算", verdict_sliding["supported"] is True,
          str(verdict_sliding["reasons"])[:70])
    check("给出滑窗提示",
          any("滑窗" in w for w in verdict_sliding["warnings"]),
          str(verdict_sliding["warnings"])[:70])

    # A vision tower is a real blocker, not a minor feature.
    multimodal = dict(sliding)
    multimodal["vision_config"] = {"hidden_size": 1152}
    verdict_mm = compatibility.check_compatibility(multimodal)
    check("多模态模型被拦截", verdict_mm["supported"] is False)

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL SUPPORT-MATRIX CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
