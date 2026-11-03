"""Build simulator YAML from UI parameters.

Two rules drive everything here:

1. `ParallelConfig.from_dict` / `HardwareConfig.from_dict` are implemented as
   `cls(**config)`, so the YAML key set must match the dataclass fields exactly
   (no missing required keys, no unknown extras).
2. Every number that feeds the estimate must come from a source the simulator
   itself ships with - the repo `config.yaml` template or the packaged operator
   profiles. Nothing here is invented.
"""

from __future__ import annotations

import copy
from typing import Any

# --------------------------------------------------------------------------
# Hardware preset - Hygon DCU BW1000.
# Values are the ones shipped in hcu-train-simulator/templates/config.yaml and
# match the defaults documented in README section 2.3.
# --------------------------------------------------------------------------
BW1000: dict[str, Any] = {
    "key": "bw1000",
    "name": "海光 DCU BW1000",
    "vendor": "Hygon",
    "use_bandwidth_table": True,
    "fp16_tflops": 480,
    "fp8_tflops": 960,
    "gpus_per_node": 8,
    "hbm_gib": 64,
    "intra_bw_gbps": 224,
    "inter_bw_gbps": 64,
    "gemm_efficiency": 0.4,
    "non_gemm_efficiency": 0.08,
    "optimizer_efficiency": 0.04,
    "p2p_intra_efficiency": 0.8,
    "collective_intra_efficiency": 0.7,
    "collective_inter_efficiency": 0.8,
    # Measured collective bandwidth tables shipped in the repo only exist for
    # bw1000 and bw500SM; estimators/communication.py hardcodes GPUType.BW1000.
    "bandwidth_source": "bw1000_1.txt / bw1000_2.txt (内置实测带宽表)",
    "notes": [
        "通信带宽来自仓库内置实测带宽表（bw1000_1.txt / bw1000_2.txt），配置中的带宽与效率字段不参与计算。",
        "estimators/communication.py 中 GPUType 固定为 BW1000。",
    ],
}

HARDWARE_PRESETS: dict[str, dict[str, Any]] = {BW1000["key"]: BW1000}

# Operator profiles packaged in hcu_train_simulator/profiles/builtin/.
# These decide per-operator compute times, which is what actually drives
# iteration time / TGS / MFU.
OPERATOR_PROFILES = [
    {
        "key": "bw1000",
        "name": "BW1000 实测算子 Profile",
        "selector": {"accelerator": "bw1000", "torch": "271", "dtk": "2604"},
        "profile_id": "bw1000_torch271_dtk2604",
        "recommended": True,
        "note": "Torch 2.7.1 + DTK 26.04 环境下实测，与 BW1000 硬件预设配套。",
    },
    {
        "key": "bw1100",
        "name": "BW1100 实测算子 Profile",
        "selector": {"accelerator": "bw1100", "torch": "271", "dtk": "2604"},
        "profile_id": "bw1100_torch271_dtk2604",
        "recommended": False,
        "note": (
            "算子 Profile 为 BW1100 实测值；通信时间仍使用仓库硬编码的 BW1000 带宽表，"
            "请按「算力取 BW1100、带宽取 BW1000」理解。"
        ),
    },
    {
        "key": "theoretical",
        "name": "理论值（不使用 Profile）",
        "selector": None,
        "profile_id": None,
        "recommended": False,
        "note": "全部算子按 gemm_efficiency / non_gemm_efficiency 理论公式计算。",
    },
]

# --------------------------------------------------------------------------
# Task classification.
# The reference simulator models full-parameter pre-training only: it has no
# LoRA and no actor/critic/reward modeling. Rather than fabricate numbers, only
# 全参 is implemented; the rest are surfaced as planned.
# --------------------------------------------------------------------------
TASKS = [
    {
        "key": "full",
        "name": "全参微调",
        "enabled": True,
        "status": "ready",
        "description": "全部权重参与训练。完整支持仓库的全部并行、显存、计算与通信模型。",
    },
    {
        "key": "pretrain",
        "name": "预训练",
        "enabled": True,
        "status": "ready",
        "description": (
            "同为全参训练，仿真算子与全参一致；预设打开全量重计算 full_recompute，"
            "并按预训练惯例使用完整序列与更大 global batch。"
        ),
    },
    {
        "key": "lora",
        "name": "LoRA 微调",
        "enabled": False,
        "status": "planned",
        "description": (
            "参考仓库未实现 LoRA 显存/计算建模（无 lora 模块、梯度仅按全参反传建模），"
            "为避免给出与参考仓库不一致的数字，暂不开放。"
        ),
    },
    {
        "key": "rl",
        "name": "强化学习",
        "enabled": False,
        "status": "planned",
        "description": (
            "参考仓库仅建模稠密/MoE 监督式训练，未建模 actor/critic、rollout 与 reward，"
            "因此暂不开放。"
        ),
    },
]

TASK_DEFAULTS = {
    # full recompute trades compute for activation memory; pretraining usually
    # accepts that because steps are far more numerous.
    "full": {"full_recompute": False, "use_distributed_optimizer": True},
    "pretrain": {"full_recompute": True, "use_distributed_optimizer": True},
}

# Fields the ParallelConfig dataclass requires (no defaults), plus the ones the
# web UI exposes. Emitting all of them keeps `cls(**config)` happy.
_PARALLEL_DEFAULTS: dict[str, Any] = {
    "tp_size": 1,
    "cp_size": 1,
    "ep_size": 1,
    "etp_size": None,
    "pp_size": 1,
    "pp_schedule": "1f1b",
    "num_layers_per_vp_stage": None,
    "use_distributed_optimizer": True,
    "decoder_first_pipeline_num_layers": None,
    "decoder_last_pipeline_num_layers": None,
    "full_recompute": False,
    "use_fp8_training": False,
    "sequence_parallel": True,
    "micro_batch_size": 1,
    "global_batch_size": 256,
    "seq_length": 4096,
    "vision_seq_length": None,
    "vision_num_images": 1,
    "num_gpus": 8,
    "overlap_mode": "auto",
    "ep_overlap_enabled": False,
    "overlap_p2p_comm": False,
    "ep_communication_backend": "alltoall",
    "tp_overlap_ratio": 0,
    "cp_overlap_ratio": 0,
    "dp_overlap_ratio": 0,
    "ep_overlap_ratio": 0,
    "etp_overlap_ratio": 0,
    "edp_overlap_ratio": 0,
    "pp_overlap_ratio": None,
    "moe_grouped_gemm": True,
    "benchmark_dtype": "bf16",
    "benchmark_device": "cuda",
    "benchmark_allow_cpu": False,
    "benchmark_warmup": 5,
    "benchmark_iters": 20,
    "benchmark_dropout_p": 0.0,
    "benchmark_causal": True,
    "benchmark_qk_norm_scale": 0.4,
    "benchmark_attention_kv_context": "local",
    "benchmark_max_local_experts": None,
}

_PROFILE_DEFAULTS = {
    "enabled": True,
    "update_on_benchmark": False,
    "require_idle": True,
    "ignore_gemm_layout": True,
    "match_permuted_gemm": True,
}

_BENCHMARK_OPS = ["all"]


def get_hardware(key: str | None) -> dict[str, Any]:
    return copy.deepcopy(HARDWARE_PRESETS.get(key or BW1000["key"], BW1000))


def get_profile(key: str | None) -> dict[str, Any]:
    for profile in OPERATOR_PROFILES:
        if profile["key"] == (key or "bw1000"):
            return copy.deepcopy(profile)
    return copy.deepcopy(OPERATOR_PROFILES[0])


def get_task(key: str | None) -> dict[str, Any]:
    for task in TASKS:
        if task["key"] == key:
            return copy.deepcopy(task)
    return copy.deepcopy(TASKS[0])


# --------------------------------------------------------------------------
# Expert-parallel legality
# --------------------------------------------------------------------------
# communication_simulator.configure_parallelism derives
#   dp  = num_gpus // tp // cp // pp
#   edp = num_gpus // ep // etp // pp
# and then asserts tp*cp*dp*pp == ep*etp*edp*pp == num_gpus. Dense models get
# etp_size=1, MoE models get etp_size=tp_size (SimulationConfig.__post_init__),
# so for MoE the legal EP values depend on TP, PP and the world size:
#   ep * etp * pp  must divide num_gpus exactly, and ep must divide the expert
# count so every rank owns a whole number of experts.

EP_CHOICES = (8, 4, 2, 1)


def valid_ep_options(
    num_gpus: int, tp_size: int, pp_size: int = 1, num_experts: int | None = None
) -> list[int]:
    """Legal EP values, largest first, for the given cluster shape."""

    etp = tp_size  # MoE default
    denominator = etp * pp_size
    if denominator <= 0 or num_gpus % denominator != 0:
        return []
    limit = num_gpus // denominator
    options = [ep for ep in EP_CHOICES if limit % ep == 0]
    if num_experts:
        options = [ep for ep in options if num_experts % ep == 0]
    return options


def explain_ep_illegal(
    num_gpus: int, tp_size: int, pp_size: int, ep_size: int, etp_size: int, num_experts: int | None
) -> str:
    etp = etp_size or tp_size
    options = valid_ep_options(num_gpus, tp_size, pp_size, num_experts)
    hint = f"当前可选 EP：{' / '.join(map(str, options))}" if options else "请调整 TP 或 PP"
    return (
        f"专家并行配置非法：EP={ep_size} × ETP={etp} × PP={pp_size} 必须整除总卡数 "
        f"{num_gpus}（MoE 模型 ETP 默认等于 TP={tp_size}）。{hint}"
    )


def build_config(
    model_path: str,
    *,
    task: str = "full",
    hardware: str = "bw1000",
    operator_profile: str = "bw1000",
    parallel: dict[str, Any] | None = None,
    hardware_overrides: dict[str, Any] | None = None,
    search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact dict shape `SimulationConfig.from_dict` expects."""

    task_cfg = get_task(task)
    if not task_cfg["enabled"]:
        raise ValueError(f"任务类型 {task!r} 尚未在参考仓库中实现")

    hw = get_hardware(hardware)
    if hardware_overrides:
        for key, value in hardware_overrides.items():
            if value is None:
                continue
            if key not in hw:
                raise ValueError(f"未知的硬件覆盖项: {key}")
            hw[key] = value

    # hardware_config is passed to HardwareConfig(**config); only real dataclass
    # fields are allowed, so the UI-facing metadata keys are dropped here.
    hardware_block = {
        "use_bandwidth_table": bool(hw["use_bandwidth_table"]),
        "fp16_tflops": float(hw["fp16_tflops"]),
        "fp8_tflops": float(hw["fp8_tflops"]),
        "gpus_per_node": int(hw["gpus_per_node"]),
        "hbm_gib": float(hw["hbm_gib"]),
        "intra_bw_gbps": float(hw["intra_bw_gbps"]),
        "inter_bw_gbps": float(hw["inter_bw_gbps"]),
        "gemm_efficiency": float(hw["gemm_efficiency"]),
        "non_gemm_efficiency": float(hw["non_gemm_efficiency"]),
        "optimizer_efficiency": float(hw["optimizer_efficiency"]),
        "p2p_intra_efficiency": float(hw["p2p_intra_efficiency"]),
        "collective_intra_efficiency": float(hw["collective_intra_efficiency"]),
        "collective_inter_efficiency": float(hw["collective_inter_efficiency"]),
    }

    parallel_block = copy.deepcopy(_PARALLEL_DEFAULTS)
    task_defaults = TASK_DEFAULTS.get(task_cfg["key"], {})
    parallel_block.update(task_defaults)
    if parallel:
        parallel_block.update({k: v for k, v in parallel.items() if v is not None})
        # Allow explicit nulls for optional fields the user cleared.
        for key in ("etp_size", "num_layers_per_vp_stage",
                    "decoder_first_pipeline_num_layers",
                    "decoder_last_pipeline_num_layers", "vision_seq_length"):
            if key in parallel and parallel[key] is None:
                parallel_block[key] = None

    parallel_block = _coerce_parallel(parallel_block, hardware_block)

    profile_cfg = get_profile(operator_profile)
    profile_block = dict(_PROFILE_DEFAULTS)
    if profile_cfg["selector"] is None:
        profile_block["mode"] = "theoretical"
        profile_block["enabled"] = False
    else:
        profile_block["mode"] = "builtin"
        profile_block["enabled"] = True
        profile_block["builtin_selector"] = dict(profile_cfg["selector"])

    search_block = {
        "enabled": False,
        "top_k": 10,
        "memory_margin_gib": 5,
        "candidates": (search or {}).get("candidates", {}),
    }

    return {
        "model_path": model_path,
        "parallel_config": parallel_block,
        "hardware_config": hardware_block,
        "profile_config": profile_block,
        "search": search_block,
    }


def _coerce_parallel(parallel: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    """Normalize types and clamp the values the simulator asserts on."""

    parallel["num_gpus"] = int(parallel["num_gpus"])
    for key in ("tp_size", "cp_size", "pp_size", "ep_size", "etp_size"):
        if parallel.get(key) is not None:
            parallel[key] = int(parallel[key])
    for key in ("micro_batch_size", "global_batch_size", "seq_length", "vision_num_images"):
        parallel[key] = int(parallel[key])
    for key in ("use_distributed_optimizer", "full_recompute", "use_fp8_training",
                "sequence_parallel", "ep_overlap_enabled", "overlap_p2p_comm",
                "moe_grouped_gemm"):
        parallel[key] = bool(parallel[key])

    # The communication topology requires num_gpus % gpus_per_node == 0.
    gpn = int(hardware["gpus_per_node"])
    if parallel["num_gpus"] < gpn:
        gpn = parallel["num_gpus"]
    hardware["gpus_per_node"] = gpn
    if parallel["num_gpus"] % gpn != 0:
        raise ValueError(
            f"num_gpus={parallel['num_gpus']} 必须是单节点卡数 gpus_per_node={gpn} 的整数倍"
        )

    num_microbatches = (
        parallel["global_batch_size"]
        / parallel["micro_batch_size"]
        / max(1, parallel["num_gpus"] // (parallel["tp_size"] * parallel["cp_size"] * parallel["pp_size"]))
    )
    if num_microbatches < 1:
        raise ValueError(
            "global_batch_size 太小：每个 DP 副本至少需要 1 个 micro batch，请增大 global_batch_size"
        )
    return parallel


def describe_derived(parallel: dict[str, Any]) -> dict[str, Any]:
    """Mirror SimulationConfig.__post_init__ derived fields for the UI."""

    tp = int(parallel["tp_size"])
    cp = int(parallel["cp_size"])
    pp = int(parallel["pp_size"])
    num_gpus = int(parallel["num_gpus"])
    model_parallel = tp * cp * pp
    dp = num_gpus // model_parallel if model_parallel else 0
    layers_per_pp = parallel.get("num_layers_per_vp_stage")
    vp_size = -1
    if layers_per_pp:
        vp_size = 1  # resolved once the model layer count is known
    return {
        "dp_size": dp,
        "model_parallel_size": model_parallel,
        "num_microbatches": (
            parallel["global_batch_size"] / parallel["micro_batch_size"] / dp if dp else 0
        ),
        "vp_size_placeholder": vp_size,
    }