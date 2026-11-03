"""Training scenario domain model.

The product supports six training modes. They are deliberately separate types
rather than one form with optional fields: the resource model, the parameter
set and the compatibility rules differ per mode, and the plan's product
principle "训练模式隔离" requires that a mode never borrows another's numbers.

`TRAINING_MODES` is the single source of truth for which modes exist, what
their display metadata is, and whether the estimator kernel can actually
model them today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrainingMode(str, Enum):
    """The six training modes in scope (plan §2)."""

    PRETRAIN = "pretrain"
    SFT_FULL = "sft_full"
    SFT_LORA = "sft_lora"
    DPO = "dpo"
    GRPO = "grpo"
    PPO = "ppo"


@dataclass(frozen=True)
class ModeSpec:
    """Static description of one training mode."""

    mode: TrainingMode
    label: str
    summary: str
    #: Whether the estimator kernel models this mode. When False the API
    #: must report "not modelled" instead of returning borrowed numbers
    #: (plan §1.3 "无模型不硬算").
    modelled: bool
    #: Why it is not modelled, shown to the user.
    unmodelled_reason: str = ""
    #: Roles the mode involves (RL only).
    roles: tuple[str, ...] = ()
    #: Extra parameter groups the UI must render for this mode.
    param_groups: tuple[str, ...] = ()
    #: A note about what the estimator can and cannot tell apart for this
    #: mode. Shown in the UI so the user is not misled into thinking two
    #: modes have different resource models when they share one kernel.
    kernel_note: str = ""


#: The simulator models full-parameter training. Pretrain and full SFT share
#: that kernel; LoRA-SFT and the three RL algorithms need resource models the
#: kernel does not have (frozen base + adapter state, or role/stage layout).
#: Shown for every mode that runs on the full-parameter training kernel.
#:
#: Measured fact, not an estimate: the simulator kernel has no pretrain/SFT
#: branch. The two modes differ only in the `full_recompute` flag, and the
#: kernel declares that field without consuming it — activation memory and
#: compute time do not branch on it. So the resource numbers for the two
#: modes are identical; what actually differs is how many steps you run and
#: how the data is laid out.
#:
#: `tests/test_task_difference.py` measures this and would fail if the
#: kernel ever starts differentiating them.
SHARED_KERNEL_NOTE = (
    "预训练与「全参数 SFT」当前共用同一套计算内核，资源估算数值一致（实测：单步时间、"
    "TGS、峰值显存均相同）——内核尚未对两者做区分建模。实际差别在于："
    "① 训练步数与总时长口径——预训练按总 Token 数折算步数，SFT 按样本数与 packing 折算；"
    "② 数据量与有效 Token 口径不同；"
    "③ 推荐的重计算策略不同——预训练步数多，通常开启全量重计算；SFT 数据量小，常用选择性重计算。"
)


MODE_SPECS: dict[TrainingMode, ModeSpec] = {
    TrainingMode.PRETRAIN: ModeSpec(
        mode=TrainingMode.PRETRAIN,
        label="预训练",
        summary="全参训练，从零开始；需要总 Token 数与训练步数",
        modelled=True,
        param_groups=("data", "optimizer", "checkpoint"),
        kernel_note=SHARED_KERNEL_NOTE,
    ),
    TrainingMode.SFT_FULL: ModeSpec(
        mode=TrainingMode.SFT_FULL,
        label="全参数 SFT",
        summary="全参监督微调；关注 packing、有效 Token 与 loss mask",
        modelled=True,
        param_groups=("data", "optimizer", "checkpoint"),
        kernel_note=SHARED_KERNEL_NOTE,
    ),
    TrainingMode.SFT_LORA: ModeSpec(
        mode=TrainingMode.SFT_LORA,
        label="LoRA-SFT",
        summary="低秩适配微调；基座冻结，只训练适配器参数",
        modelled=False,
        unmodelled_reason=(
            "模拟器内核尚未建模 LoRA：冻结基座不产生基座梯度与优化器状态，"
            "但反向传播仍需经过基座，激活与反向计算不能按可训练参数比例缩减。"
            "直接套用全参步时会产生与真实训练不一致的数字。"
        ),
        param_groups=("data", "lora", "checkpoint"),
    ),
    TrainingMode.DPO: ModeSpec(
        mode=TrainingMode.DPO,
        label="DPO",
        summary="直接偏好优化；策略模型 + 冻结参考模型，无需在线 rollout",
        modelled=False,
        unmodelled_reason=(
            "模拟器内核尚未建模 DPO：需要策略/参考双模型的显存与偏好对计算模型，"
            "当前仅有单模型全参训练的估算路径。"
        ),
        roles=("policy", "reference"),
        param_groups=("data", "dpo", "checkpoint"),
    ),
    TrainingMode.GRPO: ModeSpec(
        mode=TrainingMode.GRPO,
        label="GRPO",
        summary="组相对策略优化；含 rollout、参考/奖励与策略更新",
        modelled=False,
        unmodelled_reason=(
            "模拟器内核尚未建模 GRPO：缺少 rollout 生成、规则/奖励模型与组内相对优势计算，"
            "也没有对应的算子 Profile。"
        ),
        roles=("policy", "reference", "reward"),
        param_groups=("data", "grpo", "checkpoint"),
    ),
    TrainingMode.PPO: ModeSpec(
        mode=TrainingMode.PPO,
        label="PPO",
        summary="近端策略优化；Actor / Reference / Reward / Critic 四角色多阶段",
        modelled=False,
        unmodelled_reason=(
            "模拟器内核尚未建模 PPO：需要 Actor/Critic 双向更新、GAE 优势估计与"
            "分阶段角色放置的峰值显存模型。"
        ),
        roles=("actor", "reference", "reward", "critic"),
        param_groups=("data", "ppo", "checkpoint"),
    ),
}


def is_modelled(mode: TrainingMode) -> bool:
    return MODE_SPECS[mode].modelled


def describe_modes() -> list[dict[str, Any]]:
    """Mode catalogue for the API and the UI."""
    return [
        {
            "mode": spec.mode.value,
            "label": spec.label,
            "summary": spec.summary,
            "modelled": spec.modelled,
            "unmodelled_reason": spec.unmodelled_reason,
            "roles": list(spec.roles),
            "param_groups": list(spec.param_groups),
            "kernel_note": spec.kernel_note,
        }
        for spec in MODE_SPECS.values()
    ]


@dataclass
class ParallelPlan:
    """A validated parallel layout."""

    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    etp_size: int | None = None
    vpp_size: int = 1
    num_gpus: int = 8
    micro_batch_size: int = 1
    global_batch_size: int = 256
    seq_length: int = 4096

    @property
    def dp_size(self) -> int:
        """Data-parallel width implied by the layout.

        `dp_size = num_gpus // (tp * cp * pp)`, matching the simulator's
        own derivation (plan §5.3 requires the web layer not to duplicate
        formulas, but this one is needed to display and validate the plan).
        """
        divisor = max(1, self.tp_size * self.cp_size * self.pp_size)
        return max(1, self.num_gpus // divisor)

    @property
    def grad_accumulation(self) -> int:
        """Gradient accumulation steps: GBS = MBS × DP × accumulation.

        This floors, so it cannot be used to detect a non-divisible GBS —
        the remainder is lost. `validate_batches` checks divisibility
        separately with the exact modulus.
        """
        divisor = max(1, self.micro_batch_size * self.dp_size)
        return max(1, self.global_batch_size // divisor)

    @property
    def grad_accumulation_exact(self) -> float | None:
        """Exact gradient-accumulation factor, or None when it is not integral."""
        divisor = max(1, self.micro_batch_size * self.dp_size)
        if divisor <= 0:
            return None
        quotient = self.global_batch_size / divisor
        return quotient if quotient == int(quotient) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp_size": self.tp_size,
            "cp_size": self.cp_size,
            "pp_size": self.pp_size,
            "ep_size": self.ep_size,
            "etp_size": self.etp_size,
            "vpp_size": self.vpp_size,
            "num_gpus": self.num_gpus,
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "seq_length": self.seq_length,
            "dp_size": self.dp_size,
            "grad_accumulation": self.grad_accumulation,
        }


@dataclass
class ValidationIssue:
    """One legality problem found in a training scenario."""

    field: str
    severity: str  # "error" | "warning"
    message: str
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class TrainingScenario:
    """The normalised input of one estimate request."""

    mode: TrainingMode
    model_slug: str | None = None
    model_config_path: str | None = None
    model_revision: str | None = None
    model_config_hash: str | None = None
    hardware_id: str = "bw1000"
    num_gpus: int = 8
    parallel: ParallelPlan = field(default_factory=ParallelPlan)
    # data
    total_tokens: int | None = None
    total_samples: int | None = None
    epochs: float | None = None
    max_steps: int | None = None
    # precision / optimizer / memory
    precision: str = "bf16"
    use_distributed_optimizer: bool = True
    full_recompute: bool = False
    use_fp8: bool = False
    # checkpoint
    checkpoint_interval: int | None = None
    checkpoint_keep: int = 1
    save_optimizer_state: bool = True
    # mode specific
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)
    lora_train_embeddings: bool = False
    dpo_beta: float = 0.1
    dpo_preference_batch: int = 1
    grpo_group_size: int = 8
    grpo_rollout_length: int = 1024
    grpo_kl_coef: float = 0.0
    ppo_clip: float = 0.2
    ppo_epochs: int = 1
    ppo_gae_lambda: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "model_slug": self.model_slug,
            "model_revision": self.model_revision,
            "model_config_hash": self.model_config_hash,
            "hardware_id": self.hardware_id,
            "num_gpus": self.num_gpus,
            "parallel": self.parallel.to_dict(),
            "total_tokens": self.total_tokens,
            "total_samples": self.total_samples,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "precision": self.precision,
            "use_distributed_optimizer": self.use_distributed_optimizer,
            "full_recompute": self.full_recompute,
            "use_fp8": self.use_fp8,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_keep": self.checkpoint_keep,
            "save_optimizer_state": self.save_optimizer_state,
            "lora": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "target_modules": list(self.lora_target_modules),
                "train_embeddings": self.lora_train_embeddings,
            },
            "dpo": {"beta": self.dpo_beta, "preference_batch": self.dpo_preference_batch},
            "grpo": {
                "group_size": self.grpo_group_size,
                "rollout_length": self.grpo_rollout_length,
                "kl_coef": self.grpo_kl_coef,
            },
            "ppo": {
                "clip": self.ppo_clip,
                "epochs": self.ppo_epochs,
                "gae_lambda": self.ppo_gae_lambda,
            },
        }
