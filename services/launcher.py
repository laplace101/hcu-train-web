"""Launch-command rendering (plan G05, G06).

Turns a template + a solved plan into concrete artefacts:

  * distributed environment variables (single node and multi node)
  * volume mounts and host ports
  * the rendezvous command
  * the training YAML
  * a shell script
  * a `docker run` command

The renderer is pure: it takes a plan dict and returns strings. It never
invents numbers — every value comes from the plan the estimator produced.
"""

from __future__ import annotations

import shlex
from typing import Any

from repositories import templates as template_repo

#: Base port for the training service; the rendezvous port sits above it.
BASE_PORT = 29500
#: Offset added per node so two nodes do not collide on the same host.
NODE_PORT_OFFSET = 100


def _plan_values(plan: dict[str, Any]) -> dict[str, Any]:
    """Flatten the plan fields the templates need."""
    parallel = plan.get("parallel") or plan
    return {
        "num_gpus": parallel.get("num_gpus") or plan.get("num_gpus") or 8,
        "tp_size": parallel.get("tp_size") or 1,
        "pp_size": parallel.get("pp_size") or 1,
        "cp_size": parallel.get("cp_size") or 1,
        "ep_size": parallel.get("ep_size") or 1,
        "etp_size": parallel.get("etp_size") or 1,
        "micro_batch_size": parallel.get("micro_batch_size") or 1,
        "global_batch_size": parallel.get("global_batch_size") or 256,
        "seq_length": parallel.get("seq_length") or 4096,
        "grad_accumulation": max(
            1,
            (parallel.get("global_batch_size") or 256)
            // max(1, (parallel.get("micro_batch_size") or 1) * (parallel.get("dp_size") or 1)),
        ),
        "use_distributed_optimizer": parallel.get("use_distributed_optimizer", True),
        "full_recompute": parallel.get("full_recompute", False),
    }


def _yaml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    return shlex.quote(str(value))


def render_yaml(
    template: dict[str, Any],
    plan: dict[str, Any],
    *,
    model_path: str,
    output_dir: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Render the training YAML for a template + plan."""
    values = _plan_values(plan)
    values.update({
        "lora_r": (extra or {}).get("lora_r", 16),
        "lora_alpha": (extra or {}).get("lora_alpha", 32),
        "lora_dropout": (extra or {}).get("lora_dropout", 0.05),
        "dpo_beta": (extra or {}).get("dpo_beta", 0.1),
        "grpo_group_size": (extra or {}).get("grpo_group_size", 8),
        "ppo_clip": (extra or {}).get("ppo_clip", 0.2),
        "ppo_gae_lambda": (extra or {}).get("ppo_gae_lambda", 0.95),
        "recompute_granularity": "full" if values["full_recompute"] else "selective",
    })

    lines: list[str] = [
        f"# 由 hcu-train-web 生成 — 模板 {template['id']}",
        f"# 模板状态: {'已在目标环境验证' if template.get('verified') else '未在目标环境验证，不可直接用于生产'}",
        f"# 模型: {model_path}",
        f"# 输出目录: {output_dir}",
        "",
    ]

    # The defaults block may contain dotted keys (verl) — keep them verbatim.
    for key, raw in (template.get("defaults") or {}).items():
        value = raw
        if isinstance(raw, str) and raw.startswith("${") and raw.endswith("}"):
            token = raw[2:-1]
            value = values.get(token, raw)
        if "." in key:
            lines.append(f"{key}: {_yaml_value(value)}")
        else:
            lines.append(f"{key}: {_yaml_value(value)}")

    lines.append("")
    lines.append("model:")
    lines.append(f"  path: {_yaml_value(model_path)}")
    lines.append(f"  seq_length: {values['seq_length']}")
    lines.append("")
    lines.append("output_dir: " + _yaml_value(output_dir))

    hyperparameters = template.get("hyperparameters") or {}
    if hyperparameters:
        lines.append("")
        lines.append("hyperparameters:")
        # Templates come in two shapes: nested sections ({"optimizer": {...}})
        # and flat dotted keys (verl's "actor_rollout_ref.actor.optim.lr").
        # Both are rendered as valid YAML.
        flat = all(not isinstance(v, dict) for v in hyperparameters.values())
        for section, body in hyperparameters.items():
            if flat:
                rendered = body
                if isinstance(rendered, str) and rendered.startswith("${") and rendered.endswith("}"):
                    rendered = values.get(rendered[2:-1], rendered)
                lines.append(f"  {section}: {_yaml_value(rendered)}")
            else:
                lines.append(f"  {section}:")
                for key, value in body.items():
                    rendered = value
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        rendered = values.get(value[2:-1], value)
                    lines.append(f"    {key}: {_yaml_value(rendered)}")

    return "\n".join(lines) + "\n"


def render_env(
    plan: dict[str, Any],
    *,
    nodes: int = 1,
    node_index: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int | None = None,
) -> list[str]:
    """Render the distributed environment variables (plan G05).

    `nodes` is how many machines take part; `node_index` is this machine's
    position. Node 0 is the rendezvous master.
    """
    values = _plan_values(plan)
    gpus_per_node = values["num_gpus"] // max(1, nodes)
    # Every node talks to the node-0 master, so the rendezvous port is the
    # same everywhere; only the *local* service port is offset per node.
    port = master_port if master_port is not None else BASE_PORT

    env = [
        f"GPUS_PER_NODE={gpus_per_node}",
        f"NNODES={nodes}",
        f"NODE_RANK={node_index}",
        f"MASTER_ADDR={master_addr}",
        f"MASTER_PORT={port}",
        f"NODE_PORT={BASE_PORT + NODE_PORT_OFFSET * node_index}",
        f"WORLD_SIZE={values['num_gpus']}",
        "PYTHONPATH=${WORK_DIR}:${PYTHONPATH}",
        "PYTHONUNBUFFERED=1",
        "OMP_NUM_THREADS=8",
        "NCCL_DEBUG=WARN",
        "HSA_ENABLE_SDMA=1",
    ]
    return env


def render_rendezvous(plan: dict[str, Any], *, nodes: int = 1, master_addr: str = "127.0.0.1",
                      master_port: int = BASE_PORT) -> str:
    """The torchrun rendezvous command (plan G05)."""
    values = _plan_values(plan)
    gpus_per_node = values["num_gpus"] // max(1, nodes)
    # Shell variable references are written as "$" + "{NAME}" so the
    # f-string does not try to interpolate them.
    return (
        f"torchrun --nnodes={nodes} --nproc_per_node={gpus_per_node} "
        f"--node_rank=$" + "{NODE_RANK} --master_addr=" + master_addr + " "
        f"--master_port=$" + "{MASTER_PORT} --rdzv_backend=c10d "
        f"--rdzv_endpoint=" + f"{master_addr}:{master_port} "
        f"{values.get('entrypoint', 'train.py')}"
    )


def render_mounts(*, model_path: str, output_dir: str, data_dir: str | None = None) -> list[str]:
    """Volume mounts (plan G05)."""
    mounts = [
        f"-v {model_path}:{model_path}:ro",
        f"-v {output_dir}:{output_dir}",
    ]
    if data_dir:
        mounts.append(f"-v {data_dir}:{data_dir}:ro")
    return mounts


def render_ports(*, node_index: int = 0, extra_ports: list[int] | None = None) -> list[str]:
    """Host ports to publish (plan G05)."""
    ports = [f"-p {BASE_PORT + NODE_PORT_OFFSET * node_index}:{BASE_PORT + NODE_PORT_OFFSET * node_index}"]
    for port in extra_ports or []:
        ports.append(f"-p {port}:{port}")
    return ports


def render_shell(
    template: dict[str, Any],
    plan: dict[str, Any],
    *,
    model_path: str,
    output_dir: str,
    work_dir: str,
    image: str | None = None,
    nodes: int = 1,
    extra: dict[str, Any] | None = None,
) -> str:
    """Render a runnable shell script (plan G06)."""
    values = _plan_values(plan)
    yaml_text = render_yaml(template, plan, model_path=model_path, output_dir=output_dir, extra=extra)
    # Per-node port offset: node 0 hosts the rendezvous master.
    env = render_env(plan, nodes=nodes, node_index=0)

    lines = [
        "#!/usr/bin/env bash",
        f"# 由 hcu-train-web 生成 — 模板 {template['id']} ({template['name']})",
        "#" + ("已在目标环境验证" if template.get("verified") else "未在目标环境验证，请先在目标环境验证"),
        "set -euo pipefail",
        "",
        f"WORK_DIR={work_dir}",
        f"MODEL_PATH={model_path}",
        f"OUTPUT_DIR={output_dir}",
    ]
    if image:
        lines.append(f"IMAGE={image}")
    lines += [
        "",
        "mkdir -p \"${OUTPUT_DIR}\" \"${WORK_DIR}\"",
        "",
        "# ---- 训练配置 ----",
    ]
    for line in yaml_text.splitlines():
        lines.append(line)
    lines += [
        "",
        "# ---- 分布式环境变量 ----",
    ]
    for item in env:
        lines.append(f"export {item}")
    lines += [
        "",
        "# ---- 启动 ----",
        f'ENTRYPOINT="{template.get("entrypoint", "train.py")}"',
        "echo \"启动训练: ${ENTRYPOINT}\"",
        "${ENTRYPOINT}",
    ]
    return "\n".join(lines) + "\n"


def render_docker(
    template: dict[str, Any],
    plan: dict[str, Any],
    *,
    image: str,
    model_path: str,
    output_dir: str,
    work_dir: str,
    nodes: int = 1,
    node_index: int = 0,
    extra_ports: list[int] | None = None,
) -> str:
    """Render a `docker run` command (plan G06)."""
    parts = ["docker run --rm", "--ipc=host", "--shm-size=32g", "--network=host"]
    parts += render_ports(node_index=node_index, extra_ports=extra_ports)
    parts += render_mounts(model_path=model_path, output_dir=output_dir)
    parts += ["-w /workspace", image]
    return " \\\n  ".join(parts)


def render_all(
    template_id: str,
    plan: dict[str, Any],
    *,
    model_path: str,
    output_dir: str,
    work_dir: str,
    image: str | None = None,
    nodes: int = 1,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render every artefact for one template + plan."""
    template = template_repo.get_template(template_id)
    if template is None:
        raise KeyError(f"未知模板：{template_id}")

    values = _plan_values(plan)
    entry = dict(values)
    entry["entrypoint"] = template.get("entrypoint", "train.py")

    return {
        "template_id": template_id,
        "template_name": template["name"],
        "verified": template.get("verified", False),
        "verified_on": template.get("verified_on"),
        "notes": template.get("notes"),
        "image": image,
        "yaml": render_yaml(template, plan, model_path=model_path, output_dir=output_dir, extra=extra),
        "shell": render_shell(
            template, plan, model_path=model_path, output_dir=output_dir,
            work_dir=work_dir, image=image, nodes=nodes, extra=extra,
        ),
        "docker": render_docker(
            template, plan, image=image or "<image>", model_path=model_path,
            output_dir=output_dir, work_dir=work_dir, nodes=nodes,
        ),
        "env": render_env(plan, nodes=nodes),
        "rendezvous": render_rendezvous(plan, nodes=nodes),
    }
