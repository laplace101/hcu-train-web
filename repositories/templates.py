"""Training configuration templates and launch-command rendering (plan G03–G06).

A template describes *how* to launch a training run for one mode on one
framework. It is deliberately data, not code: the renderer turns a template
plus a solved plan into YAML / shell / docker commands.

Templates carry a `verified` flag. Only templates verified on a real Hygon
environment may be rendered as directly runnable; unverified ones are still
offered but are clearly labelled, per the plan's exit condition for stage G
("未验证模板不会误导为可直连运行").
"""

from __future__ import annotations

from typing import Any

#: Template catalogue. `id` is stable and referenced by the API.
TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "megatron-pretrain",
        "mode": "pretrain",
        "name": "Megatron-LM 预训练",
        "framework": "megatron",
        "stack_id": "megatron-dtk26-py310",
        "entrypoint": "torchrun --nproc_per_node=${GPUS_PER_NODE} pretrain_gpt.py",
        "verified": True,
        "verified_on": "BW1000 / DTK 26.04 / 8 卡",
        "notes": "张量并行 + 流水并行的标准预训练入口。",
        "defaults": {
            "tensor_model_parallel_size": "${tp_size}",
            "pipeline_model_parallel_size": "${pp_size}",
            "context_parallel_size": "${cp_size}",
            "expert_model_parallel_size": "${ep_size}",
            "micro_batch_size": "${micro_batch_size}",
            "global_batch_size": "${global_batch_size}",
            "seq_length": "${seq_length}",
            "recompute_granularity": "${recompute_granularity}",
            "use_distributed_optimizer": "${use_distributed_optimizer}",
        },
        "hyperparameters": {
            "optimizer": {"type": "adam", "lr": 1.0e-4, "min_lr": 1.0e-5, "weight_decay": 0.1},
            "schedule": {"type": "cosine", "warmup_iters": 2000},
        },
    },
    {
        "id": "megatron-sft-full",
        "mode": "sft_full",
        "name": "Megatron-LM 全参 SFT",
        "framework": "megatron",
        "stack_id": "megatron-dtk26-py310",
        "entrypoint": "torchrun --nproc_per_node=${GPUS_PER_NODE} sft_gpt.py",
        "verified": True,
        "verified_on": "BW1000 / DTK 26.04 / 8 卡",
        "notes": "在预训练权重基础上做全参数微调。",
        "defaults": {
            "tensor_model_parallel_size": "${tp_size}",
            "pipeline_model_parallel_size": "${pp_size}",
            "context_parallel_size": "${cp_size}",
            "expert_model_parallel_size": "${ep_size}",
            "micro_batch_size": "${micro_batch_size}",
            "global_batch_size": "${global_batch_size}",
            "seq_length": "${seq_length}",
            "use_distributed_optimizer": "${use_distributed_optimizer}",
        },
        "hyperparameters": {
            "optimizer": {"type": "adam", "lr": 1.0e-5, "min_lr": 1.0e-6, "weight_decay": 0.1},
            "schedule": {"type": "cosine", "warmup_iters": 50},
        },
    },
    {
        "id": "deepspeed-sft-full",
        "mode": "sft_full",
        "name": "DeepSpeed ZeRO-3 全参微调",
        "framework": "deepspeed",
        "stack_id": "deepspeed-dtk26-py310",
        "entrypoint": "deepspeed --num_gpus=${NUM_GPUS} train.py --deepspeed ds_config.json",
        "verified": True,
        "verified_on": "BW1000 / DTK 26.04 / 8 卡",
        "notes": "ZeRO-3 分片优化器状态，适合单模型放不下的场景。",
        "defaults": {
            "train_micro_batch_size_per_gpu": "${micro_batch_size}",
            "gradient_accumulation_steps": "${grad_accumulation}",
            "train_batch_size": "${global_batch_size}",
            "model_seq_length": "${seq_length}",
        },
        "hyperparameters": {
            "optimizer": {"type": "adamw", "lr": 1.0e-5},
            "schedule": {"type": "cosine", "warmup_ratio": 0.03},
        },
    },
    {
        "id": "llamafactory-lora-sft",
        "mode": "sft_lora",
        "name": "LLaMA-Factory LoRA 微调",
        "framework": "llamafactory",
        "stack_id": "llamafactory-dtk25-py310",
        "entrypoint": "llamafactory-cli train ${WORK_DIR}/train_config.yaml",
        "verified": True,
        "verified_on": "BW1000 / DTK 25.04 / 8 卡",
        "notes": "LoRA 低秩微调；显存与全参训练差异显著，不可用全参估算代替。",
        "defaults": {
            "finetuning_type": "lora",
            "lora_rank": "${lora_r}",
            "lora_alpha": "${lora_alpha}",
            "lora_dropout": "${lora_dropout}",
            "per_device_train_batch_size": "${micro_batch_size}",
            "gradient_accumulation_steps": "${grad_accumulation}",
            "cutoff_len": "${seq_length}",
        },
        "hyperparameters": {
            "optimizer": {"type": "adamw_torch", "lr": 1.0e-4},
            "schedule": {"type": "cosine", "warmup_ratio": 0.1},
        },
    },
    {
        "id": "llamafactory-dpo",
        "mode": "dpo",
        "name": "LLaMA-Factory DPO 偏好对齐",
        "framework": "llamafactory",
        "stack_id": "llamafactory-dtk25-py310",
        "entrypoint": "llamafactory-cli train ${WORK_DIR}/train_config.yaml",
        "verified": False,
        "verified_on": None,
        "notes": "需要同时前向参考模型；本模板尚未在目标环境验证。",
        "defaults": {
            "finetuning_type": "lora",
            "pref_beta": "${dpo_beta}",
            "pref_ftx": 0.1,
            "per_device_train_batch_size": "${micro_batch_size}",
            "cutoff_len": "${seq_length}",
        },
        "hyperparameters": {
            "optimizer": {"type": "adamw_torch", "lr": 5.0e-6},
        },
    },
    {
        "id": "verl-grpo",
        "mode": "grpo",
        "name": "verl GRPO 强化学习",
        "framework": "verl",
        "stack_id": "verl-dtk26-py310",
        "entrypoint": "python3 -m verl.trainer.main_ppo actor_rollout_ref.rollout=native",
        "verified": False,
        "verified_on": None,
        "notes": "包含 rollout 与训练双阶段，资源模型尚未建立；本模板尚未在目标环境验证。",
        "defaults": {
            "actor_rollout_ref.rollout.n": "${grpo_group_size}",
            "data.max_prompt_length": "${seq_length}",
            "trainer.n_gpus_per_node": "${NUM_GPUS}",
        },
        "hyperparameters": {
            "actor_rollout_ref.actor.optim.lr": 1.0e-6,
            "algorithm.adv_estimator": "grpo",
        },
    },
    {
        "id": "verl-ppo",
        "mode": "ppo",
        "name": "verl PPO 强化学习",
        "framework": "verl",
        "stack_id": "verl-dtk26-py410",
        "entrypoint": "python3 -m verl.trainer.main_ppo actor_rollout_ref.rollout=native",
        "verified": False,
        "verified_on": None,
        "notes": "包含 actor/critic 双模型；本模板尚未在目标环境验证。",
        "defaults": {
            "actor_rollout_ref.rollout.n": "${ppo_clip}",
            "data.max_prompt_length": "${seq_length}",
            "trainer.n_gpus_per_node": "${NUM_GPUS}",
        },
        "hyperparameters": {
            "algorithm.adv_estimator": "gae",
            "algorithm.gae_lambda": "${ppo_gae_lambda}",
        },
    },
]


def list_templates(mode: str | None = None, *, verified_only: bool = False) -> list[dict[str, Any]]:
    out = []
    for template in TEMPLATES:
        if mode and template["mode"] != mode:
            continue
        if verified_only and not template["verified"]:
            continue
        out.append(template)
    return out


def get_template(template_id: str) -> dict[str, Any] | None:
    for template in TEMPLATES:
        if template["id"] == template_id:
            return template
    return None


def templates_for_mode(mode: str) -> list[dict[str, Any]]:
    """Templates available for a mode, with the verified one first.

    A mode whose kernel is not modelled still gets its templates listed —
    the templates are real, it is the *estimates* that are unavailable.
    """
    matched = list_templates(mode)
    matched.sort(key=lambda t: (not t["verified"], t["name"]))
    return matched
