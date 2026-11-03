"""Checks for the image catalogue, templates and launch rendering (plan G).

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_launch.py
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


PLAN = {
    "num_gpus": 8, "tp_size": 2, "pp_size": 1, "cp_size": 1, "ep_size": 1,
    "micro_batch_size": 1, "global_batch_size": 256, "seq_length": 4096, "dp_size": 4,
}


def main() -> int:
    client = app.test_client()

    print("=== ① 镜像目录 ===")
    r = client.get("/api/v1/images")
    body = r.get_json()
    check("images 200", r.status_code == 200 and body["ok"])
    check("有可用训练镜像", body["count"] > 0, f"{body['count']} 个仓库")
    check("标注了框架分布", len(body["frameworks"]) > 0,
          ", ".join(f"{f['framework']}×{f['tag_count']}" for f in body["frameworks"][:5]))
    check("给出镜像来源", bool(body["source"]["path"]), body["source"]["path"] or "")
    tags = body["images"][0]["tags"]
    check("tag 含 pull_url", all(t["pull_url"] for t in tags), tags[0]["pull_url"][:60])
    check("tag 解析出 DTK 版本", any(t["dtk_version"] for t in tags),
          next((t["dtk_version"] for t in tags if t["dtk_version"]), ""))

    print("\n=== ② 提取码脱敏（G02）===")
    r = client.post("/api/v1/images/codes", json={"repo": "admin/base/custom", "code": "SECRET123"})
    body = r.get_json()
    check("存储成功", r.status_code == 200 and body["ok"])
    check("不回显明文", "SECRET123" not in str(body), str(body))
    r = client.get("/api/v1/images/codes")
    listing = r.get_json()["codes"]
    entry = next((c for c in listing if c["repo"] == "admin/base/custom"), None)
    check("列表项不含明文", entry is not None and "SECRET123" not in str(entry), str(entry))
    check("列表项有掩码提示", entry is not None and entry.get("code_hint") == "••••")
    r = client.post("/api/v1/images/codes/verify",
                    json={"repo": "admin/base/custom", "code": "SECRET123"})
    check("正确码校验通过", r.get_json()["matches"] is True)
    r = client.post("/api/v1/images/codes/verify",
                    json={"repo": "admin/base/custom", "code": "WRONG"})
    check("错误码校验失败", r.get_json()["matches"] is False)

    print("\n=== ③ 训练模板（G03/G04）===")
    r = client.get("/api/v1/templates")
    templates = r.get_json()["templates"]
    modes = {t["mode"] for t in templates}
    check("覆盖六种模式", modes == {"pretrain", "sft_full", "sft_lora", "dpo", "grpo", "ppo"},
          str(sorted(modes)))
    check("未验证模板有标注",
          all(("verified_on" in t) for t in templates),
          f"{sum(1 for t in templates if t['verified'])} 个已验证 / {len(templates)} 个")
    r = client.get("/api/v1/templates?mode=grpo")
    check("按模式过滤", all(t["mode"] == "grpo" for t in r.get_json()["templates"]))

    print("\n=== ④ 命令渲染（G05/G06）===")
    r = client.post("/api/v1/launch", json={
        "template_id": "megatron-sft-full", "plan": PLAN,
        "model_path": "/models/Qwen3-8B", "output_dir": "/workspace/out",
        "work_dir": "/workspace",
        "image": "harbor.sourcefind.cn:5443/hcu/admin/base/custom:dtk26.04-py3.10",
        "nodes": 1,
    })
    body = r.get_json()
    check("launch 200", r.status_code == 200 and body["ok"], str(r.status_code))
    if body.get("ok"):
        yaml_text = body["yaml"]
        check("YAML 含并行参数", "tensor_model_parallel_size: 2" in yaml_text)
        check("YAML 含 batch 参数", "global_batch_size: 256" in yaml_text)
        check("YAML 含模型路径", "/models/Qwen3-8B" in yaml_text)
        check("YAML 标注验证状态", "已在目标环境验证" in yaml_text)
        shell = body["shell"]
        check("Shell 含 shebang", shell.startswith("#!/usr/bin/env bash"))
        check("Shell 含 WORLD_SIZE", "WORLD_SIZE=8" in shell)
        check("Shell 含分布式变量", "MASTER_ADDR=" in shell and "MASTER_PORT=" in shell)
        check("Shell 含 DCU 环境变量", "HSA_ENABLE_SDMA=1" in shell)
        check("Docker 含镜像", "dtk26.04-py3.10" in body["docker"])
        check("Docker 含挂载", "/models/Qwen3-8B" in body["docker"])
        check("rendezvous 用 torchrun", body["rendezvous"].startswith("torchrun"))
        check("rendezvous 带 shell 变量", "${NODE_RANK}" in body["rendezvous"])
        check("rendezvous 不含 Python 插值错误", "NameError" not in body["rendezvous"])

    print("\n=== ⑤ 多机渲染 ===")
    r = client.post("/api/v1/launch", json={
        "template_id": "deepspeed-sft-full", "plan": PLAN,
        "model_path": "/models/Qwen3-8B", "output_dir": "/workspace/out",
        "work_dir": "/workspace", "image": "img:tag", "nodes": 2,
    })
    body = r.get_json()
    if body.get("ok"):
        env = body["env"]
        check("多机 NNODES=2", "NNODES=2" in env)
        check("每节点卡数正确", "GPUS_PER_NODE=4" in env,
              next((e for e in env if e.startswith("GPUS_PER_NODE")), ""))
        # All nodes talk to the node-0 master, so MASTER_PORT is identical
        # everywhere; the per-node offset applies to the local service port.
        check("MASTER_PORT 全节点一致", "MASTER_PORT=29500" in env,
              next((e for e in env if "MASTER_PORT" in e), ""))
        check("本地服务端口按节点错开", "NODE_PORT=29500" in env,
              next((e for e in env if "NODE_PORT" in e), ""))

    print("\n=== ⑥ LoRA 模板使用专属参数 ===")
    r = client.post("/api/v1/launch", json={
        "template_id": "llamafactory-lora-sft", "plan": PLAN,
        "model_path": "/models/Qwen3-8B", "output_dir": "/workspace/out",
        "work_dir": "/workspace", "image": "img:tag",
        "extra": {"lora_r": 32, "lora_alpha": 64},
    })
    body = r.get_json()
    if body.get("ok"):
        check("YAML 含 lora_rank", "lora_rank: 32" in body["yaml"])
        check("YAML 含 lora_alpha", "lora_alpha: 64" in body["yaml"])
        check("不是全参模板", "tensor_model_parallel_size" not in body["yaml"])

    print("\n=== ⑦ 未验证模板明确标注 ===")
    r = client.post("/api/v1/launch", json={
        "template_id": "verl-grpo", "plan": PLAN,
        "model_path": "/models/Qwen3-8B", "output_dir": "/workspace/out",
        "work_dir": "/workspace", "image": "img:tag",
    })
    body = r.get_json()
    if body.get("ok"):
        check("verified=False", body["verified"] is False)
        check("YAML 标注未验证", "未在目标环境验证" in body["yaml"])

    print("\n=== ⑧ 错误处理 ===")
    r = client.post("/api/v1/launch", json={"template_id": "nope", "plan": PLAN})
    check("未知模板报错", r.status_code == 400
          and r.get_json()["error"]["code"] == "unknown_template")
    r = client.post("/api/v1/launch", json={"plan": PLAN})
    check("缺 template_id 报错", r.status_code == 400)

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL LAUNCH/IMAGE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
