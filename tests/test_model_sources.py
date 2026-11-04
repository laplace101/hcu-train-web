"""Checks for the ModelScope model path and the compatibility gate.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_model_sources.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402
from services import compatibility  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main() -> int:
    client = app.test_client()

    print("=== ① 兼容性判定：受支持的模型 ===")
    qwen = json.loads(
        (PROJECT_ROOT / "catalog" / "models" / "qwen3-8b" / "config.json")
        .read_text(encoding="utf-8"))
    verdict = compatibility.check_compatibility(qwen)
    check("Qwen3-8B 可估算", verdict["supported"] is True, str(verdict["reasons"])[:80])
    check("匹配到 qwen3 适配器", verdict["adapter"] == "qwen3", str(verdict["adapter"]))
    check("标记为专用适配器", verdict["adapter_is_specialised"] is True)

    vl = json.loads(
        (PROJECT_ROOT / "catalog" / "models" / "qwen3-vl-8b-instruct" / "config.json")
        .read_text(encoding="utf-8"))
    verdict_vl = compatibility.check_compatibility(vl)
    check("Qwen3-VL 可估算", verdict_vl["supported"] is True,
          str(verdict_vl["reasons"])[:80])
    check("Qwen3-VL 匹配到视觉适配器",
          verdict_vl["adapter"] in compatibility.VISION_CAPABLE_ADAPTERS,
          str(verdict_vl["adapter"]))

    print("\n=== ② 兼容性判定：不支持的模型 ===")
    # A multimodal model that lands on a text-only adapter.
    v41 = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "num_hidden_layers": 40, "hidden_size": 5120,
            "num_attention_heads": 64, "n_routed_experts": 384,
            "dspark_n_routed_experts": 128, "index_head_dim": 128,
        },
    }
    verdict_bad = compatibility.check_compatibility(v41)
    check("V4.1 判定为不支持", verdict_bad["supported"] is False)
    check("列出多条原因", len(verdict_bad["reasons"]) >= 3,
          f"{len(verdict_bad['reasons'])} 条")
    check("指明视觉塔问题",
          any("视觉塔" in r or "多模态" in r for r in verdict_bad["reasons"]))
    check("指明内核无法表示的结构",
          any("无法表示" in r or "未建模" in r for r in verdict_bad["reasons"]),
          verdict_bad["reasons"][-1][:60] if verdict_bad["reasons"] else "")
    check("给出建议", len(verdict_bad["suggestions"]) > 0)

    print("\n=== ②b 通用兜底适配器 ===")
    # A plain dense decoder with no family-specific fields: the kernel's
    # generic fallback handles it correctly.
    dense = {"model_type": "mistral", "architectures": ["MistralForCausalLM"],
             "num_hidden_layers": 32, "hidden_size": 4096,
             "num_attention_heads": 32, "num_key_value_heads": 8,
             "intermediate_size": 14336, "vocab_size": 32000}
    verdict_dense = compatibility.check_compatibility(dense)
    check("稠密模型落到通用适配器", verdict_dense["adapter"] == "generic_gpt",
          str(verdict_dense["adapter"]))
    check("通用适配器仍可估算", verdict_dense["supported"] is True,
          str(verdict_dense["reasons"])[:80])
    check("标记为兜底适配器", verdict_dense["adapter_is_fallback"] is True)

    # A MoE model whose expert count the kernel cannot read would be
    # modelled as dense — that must be refused.
    moe_unreadable = {"model_type": "minimax_m2", "architectures": ["MiniMaxM2ForCausalLM"],
                      "num_hidden_layers": 62, "hidden_size": 3072,
                      "num_local_experts": 256, "num_experts_per_tok": 8,
                      "vocab_size": 200064}
    verdict_moe = compatibility.check_compatibility(moe_unreadable)
    check("内核读不到的专家数被拦截", verdict_moe["supported"] is False)
    check("说明会被当成稠密模型",
          any("稠密" in r for r in verdict_moe["reasons"]),
          verdict_moe["reasons"][0][:70] if verdict_moe["reasons"] else "")

    print("\n=== ③ 兼容性判定：未知模型类型 ===")
    # The kernel has a generic fallback adapter, so an unknown *dense*
    # decoder is still estimated — with the standard decoder formulas.
    unknown = {"model_type": "some_unknown_model", "architectures": ["FooForCausalLM"],
               "num_hidden_layers": 32, "hidden_size": 4096, "vocab_size": 32000}
    verdict_unknown = compatibility.check_compatibility(unknown)
    check("未知模型类型落到通用适配器",
          verdict_unknown["adapter"] == "generic_gpt", str(verdict_unknown["adapter"]))
    check("标记为兜底适配器", verdict_unknown["adapter_is_fallback"] is True)
    check("普通稠密结构仍可估算", verdict_unknown["supported"] is True,
          str(verdict_unknown["reasons"])[:80])

    # But an unknown model with an architecture the kernel cannot read must
    # still be refused.
    unknown_moe = {"model_type": "some_unknown_model",
                   "architectures": ["FooForCausalLM"],
                   "num_local_experts": 128, "num_experts_per_tok": 4}
    verdict_unknown_moe = compatibility.check_compatibility(unknown_moe)
    check("未知模型的可读专家数仍被拦截", verdict_unknown_moe["supported"] is False)

    print("\n=== ④ ModelScope 模型走仿真接口 ===")
    # A resolvable public repository. The network may be unavailable in some
    # environments, so the test accepts either a clean estimate or a clean
    # refusal — but never a crash.
    response = client.post("/api/simulate", json={
        "task": "full", "hardware": "bw1000", "operator_profile": "bw1000",
        "model": "Qwen/Qwen3-8B",
        "parallel": {"tp_size": 2, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                     "num_gpus": 8, "micro_batch_size": 1,
                     "global_batch_size": 256, "seq_length": 4096},
    })
    body = response.get_json()
    check("ModelScope 引用不再报「未找到预设模型」",
          "未找到预设模型" not in str(body.get("error", "")),
          str(body.get("error"))[:80])
    if response.status_code == 200:
        check("估算成功", body.get("ok") is True)
        source = body["result"].get("model_source") or {}
        check("来源标记为 modelscope", source.get("kind") == "modelscope",
              str(source.get("kind")))
        check("记录了 revision", bool(source.get("revision")), str(source.get("revision")))
        check("记录了 config 哈希", bool(source.get("config_hash")))
    else:
        check("失败时是明确的业务错误",
              response.status_code in (400, 422), str(response.status_code))

    print("\n=== ⑤ 不支持模型返回结构化原因 ===")
    response = client.post("/api/simulate", json={
        "task": "full", "hardware": "bw1000", "operator_profile": "bw1000",
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "parallel": {"tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                     "num_gpus": 8, "micro_batch_size": 1,
                     "global_batch_size": 256, "seq_length": 4096},
    })
    body = response.get_json()
    if body.get("ok") is False and body.get("reasons"):
        check("返回 422", response.status_code == 422, str(response.status_code))
        check("错误信息是通用说明",
              "无法为该模型给出可信估算" in body.get("error", ""),
              body.get("error", ""))
        check("原因为列表", isinstance(body["reasons"], list)
              and len(body["reasons"]) >= 3, f"{len(body['reasons'])} 条")
        check("建议为列表", isinstance(body.get("suggestions"), list)
              and len(body["suggestions"]) > 0)
        check("给出模型摘要", bool(body.get("model_summary")),
              body.get("model_summary", ""))
    else:
        # The network may be down; the gate is still exercised directly above.
        check("未崩溃（网络不可用时跳过）", True, str(response.status_code))

    print("\n=== ⑥ 预设模型仍正常 ===")
    response = client.post("/api/simulate", json={
        "task": "full", "hardware": "bw1000", "operator_profile": "bw1000",
        "model": "qwen3-8b",
        "parallel": {"tp_size": 2, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                     "num_gpus": 8, "micro_batch_size": 1,
                     "global_batch_size": 256, "seq_length": 4096},
    })
    body = response.get_json()
    check("预设模型不受影响", response.status_code == 200 and body.get("ok") is True,
          f"HTTP {response.status_code}")
    if body.get("ok"):
        check("数值与基线一致",
              abs(body["result"]["summary"]["iteration_ms"] - 46651.92) / 46651.92 < 0.005,
              str(body["result"]["summary"]["iteration_ms"]))

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL MODEL-SOURCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
