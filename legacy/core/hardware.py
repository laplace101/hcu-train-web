"""Built-in hardware presets (peak numbers + efficiency tuning knobs).

`sources` field traces where each number came from. These presets are referenced
by both the UI (frontend) and the estimator (compute + memory).
"""


PRESETS = [
    {
        "key": "hygon_dcu_bw1000",
        "label": "Hygon DCU BW1000 (Z100/K100) 理论峰值",
        "fp16_tflops": 480,
        "fp8_tflops": 960,
        "hbm_gib": 64,
        "intra_bw_gbps": 224,
        "inter_bw_gbps": 64,
        "gpus_per_node": 8,
        "gemm_efficiency": 0.40,
        "non_gemm_efficiency": 0.08,
        "optimizer_efficiency": 0.04,
        "p2p_intra_efficiency": 0.80,
        "collective_intra_efficiency": 0.70,
        "collective_inter_efficiency": 0.80,
        "notes": "BW1000 规格; gemm_eff=0.40 为 Hygon/AMD 通用经验值, 与 hcu-train-simulator 默认对齐.",
    },
    {
        "key": "hygon_dcu_bw1000_empirical",
        "label": "Hygon DCU BW1000 (DCU 实测: GLM-4.6V-Flash 9B)",
        "fp16_tflops": 220,
        "fp8_tflops": 440,
        "hbm_gib": 64,
        "intra_bw_gbps": 224,
        "inter_bw_gbps": 64,
        "gpus_per_node": 8,
        "gemm_efficiency": 0.15,
        "non_gemm_efficiency": 0.06,
        "optimizer_efficiency": 0.04,
        "p2p_intra_efficiency": 0.80,
        "collective_intra_efficiency": 0.70,
        "collective_inter_efficiency": 0.80,
        "notes": "GLM-4.6V-Flash 9B 4 卡实测 BF16 step time ~10.79 s, 反算 gemm_eff ~0.15. 其它同架构模型可参考. "\
                 "与理论峰值的差距主要来自 inductor / HCU fused kernels 性能.",
    },
    {
        "key": "hygon_dcu_bw1100",
        "label": "Hygon DCU BW1100 (K100-Pro/Z100L) 理论峰值",
        "fp16_tflops": 700,
        "fp8_tflops": 1400,
        "hbm_gib": 80,
        "intra_bw_gbps": 448,
        "inter_bw_gbps": 200,
        "gpus_per_node": 8,
        "gemm_efficiency": 0.40,
        "non_gemm_efficiency": 0.08,
        "optimizer_efficiency": 0.04,
        "p2p_intra_efficiency": 0.85,
        "collective_intra_efficiency": 0.75,
        "collective_inter_efficiency": 0.85,
        "notes": "BW1100 理论峰值; gemm_eff 沿用同源经验.",
    },
    {
        "key": "nvidia_a100",
        "label": "NVIDIA A100 80GB (SXM)",
        "fp16_tflops": 312,
        "fp8_tflops": 624,
        "hbm_gib": 80,
        "intra_bw_gbps": 600,
        "inter_bw_gbps": 25,
        "gpus_per_node": 8,
        "gemm_efficiency": 0.40,
        "non_gemm_efficiency": 0.08,
        "optimizer_efficiency": 0.04,
        "p2p_intra_efficiency": 0.85,
        "collective_intra_efficiency": 0.75,
        "collective_inter_efficiency": 0.80,
        "notes": "A100 标准参数. NVLink 600 GB/s; IB 25 GB/s 默认.",
    },
    {
        "key": "nvidia_h100",
        "label": "NVIDIA H100 80GB (SXM5)",
        "fp16_tflops": 989,
        "fp8_tflops": 1979,
        "hbm_gib": 80,
        "intra_bw_gbps": 900,
        "inter_bw_gbps": 50,
        "gpus_per_node": 8,
        "gemm_efficiency": 0.45,
        "non_gemm_efficiency": 0.10,
        "optimizer_efficiency": 0.05,
        "p2p_intra_efficiency": 0.90,
        "collective_intra_efficiency": 0.80,
        "collective_inter_efficiency": 0.85,
        "notes": "H100 SXM5 BF16 tensor core 989 TFLOPS @ sparsity off. NVLink 900 GB/s.",
    },
    {
        "key": "nvidia_h200",
        "label": "NVIDIA H200 141GB (SXM)",
        "fp16_tflops": 989,
        "fp8_tflops": 1979,
        "hbm_gib": 141,
        "intra_bw_gbps": 900,
        "inter_bw_gbps": 50,
        "gpus_per_node": 8,
        "gemm_efficiency": 0.45,
        "non_gemm_efficiency": 0.10,
        "optimizer_efficiency": 0.05,
        "p2p_intra_efficiency": 0.90,
        "collective_intra_efficiency": 0.80,
        "collective_inter_efficiency": 0.85,
        "notes": "H200 升级主要在 HBM 141GB; TFLOPS 与 H100 持平.",
    },
]


def get_preset(name):
    for p in PRESETS:
        if p["key"] == name:
            return dict(p)
    raise ValueError("Unknown hardware preset: " + name)


def list_presets():
    return [dict(p) for p in PRESETS]
