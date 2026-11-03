"""Backend for /root/hcu-train-web.

Talks to the hcu-train-simulator CLI by writing a YAML config and invoking
hcu-train-simulator's cli in a subprocess under Python 3.14 (the uv-installed
CPython that supports PEP 604 / 585 / importlib.resources.files).

Endpoints:
  GET  /api/health       probe Python interpreter status
  GET  /api/presets      enumerate model + hardware presets
  POST /api/predict      run simulator on a model config

Run:
    python3 /root/hcu-train-web/backend.py            # binds 127.0.0.1:7766
    PORT=8080 HOST=0.0.0.0 python3 .../backend.py    # override
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
# Allow override of the simulator repo dir via env var.
REPO_DIR = Path(
    os.environ.get("HCU_SIM_REPO", "/root/hcu-train-simulator")
)
# Candidate Python interpreters (in preference order).
_PY_CANDIDATES = [
    "/root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14",
    "/root/.hermes/tools/python-3.14.7+20260901-linux-x64/bin/python3.14",
    "/opt/python310/bin/python3.10",
]

# --- Locate a usable Python interpreter ---------------------------------------

def _resolve_python():
    for p in _PY_CANDIDATES:
        if Path(p).exists():
            return p
    return sys.executable


# --- Preset catalog ----------------------------------------------------------

sys.path.insert(0, str(HERE))
from hardware_presets import PRESETS as HW_PRESETS  # noqa: E402

TASKS = [
    {
        "key": "full_ft",
        "label": "全参微调 (full fine-tune)",
        "hint": "全模型参数都训练。最常见的 fine-tune 模式。",
    },
    {
        "key": "pretrain",
        "label": "预训练 (pretrain)",
        "hint": "从头训练。用更大 batch 与 pp schedule；output 层/loss mask 与 SFT 略有不同。",
    },
    {
        "key": "lora",
        "label": "LoRA 微调 (lora)",
        "hint": "⚠ 暂未实装 LoRA 计算，目前与全参使用同一仿真器，结果仅作 full-FT 参考。",
    },
]


MODEL_PRESETS = [

    # ---- deepseek_v3 (2 models) ----
    {"key": "deepseek_v3", "label": "DeepSeek-V3 (MoE 671B)", "model_path": "deepseek-ai/DeepSeek-V3", "rough_params_B": 671.0, "kind": "moe_decoder", "adapter": "deepseek_v3"},
    {"key": "deepseek_v2", "label": "DeepSeek-V2 (MoE 236B)", "model_path": "deepseek-ai/DeepSeek-V2", "rough_params_B": 236.0, "kind": "moe_decoder", "adapter": "deepseek_v3"},
    # ---- generic_gpt (12 models) ----
    {"key": "deepseek_llm_7b", "label": "DeepSeek-LLM 7B (Llama arch)", "model_path": "deepseek-ai/deepseek-llm-7b-base", "rough_params_B": 7.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "deepseek_llm_67b", "label": "DeepSeek-LLM 67B (Llama arch)", "model_path": "deepseek-ai/deepseek-llm-67b-base", "rough_params_B": 67.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "llama3_8b", "label": "LLaMA-3-8B", "model_path": "meta-llama/Meta-Llama-3-8B", "rough_params_B": 8.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "llama3_70b", "label": "LLaMA-3-70B", "model_path": "meta-llama/Meta-Llama-3-70B", "rough_params_B": 70.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "llama3_1_8b", "label": "LLaMA-3.1-8B", "model_path": "meta-llama/Llama-3.1-8B", "rough_params_B": 8.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "llama3_1_70b", "label": "LLaMA-3.1-70B", "model_path": "meta-llama/Llama-3.1-70B", "rough_params_B": 70.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "llama3_2_1b", "label": "LLaMA-3.2-1B", "model_path": "meta-llama/Llama-3.2-1B", "rough_params_B": 1.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "llama3_2_3b", "label": "LLaMA-3.2-3B", "model_path": "meta-llama/Llama-3.2-3B", "rough_params_B": 3.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "qwen2_5_7b", "label": "Qwen2.5-7B", "model_path": "Qwen/Qwen2.5-7B", "rough_params_B": 7.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "qwen2_5_72b", "label": "Qwen2.5-72B", "model_path": "Qwen/Qwen2.5-72B", "rough_params_B": 72.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "gemma2_9b", "label": "Gemma-2-9B", "model_path": "google/gemma-2-9b", "rough_params_B": 9.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    {"key": "phi3_medium", "label": "Phi-3-medium-4k-instruct", "model_path": "microsoft/Phi-3-medium-4k-instruct", "rough_params_B": 14.0, "kind": "dense_decoder", "adapter": "generic_gpt"},
    # ---- glm5 (1 models) ----
    {"key": "glm5_moe_dsa", "label": "GLM-5 (MoE + DSA)", "model_path": "THUDM/GLM-5", "rough_params_B": 744.0, "kind": "moe_decoder", "adapter": "glm5"},
    # ---- qwen3 (8 models) ----
    {"key": "qwen3_0_6b", "label": "Qwen3-0.6B", "model_path": "Qwen/Qwen3-0.6B", "rough_params_B": 0.6, "kind": "dense_decoder", "adapter": "qwen3"},
    {"key": "qwen3_1_7b", "label": "Qwen3-1.7B", "model_path": "Qwen/Qwen3-1.7B", "rough_params_B": 1.7, "kind": "dense_decoder", "adapter": "qwen3"},
    {"key": "qwen3_4b", "label": "Qwen3-4B", "model_path": "Qwen/Qwen3-4B", "rough_params_B": 4.0, "kind": "dense_decoder", "adapter": "qwen3"},
    {"key": "qwen3_8b", "label": "Qwen3-8B", "model_path": "Qwen/Qwen3-8B", "rough_params_B": 8.0, "kind": "dense_decoder", "adapter": "qwen3"},
    {"key": "qwen3_14b", "label": "Qwen3-14B", "model_path": "Qwen/Qwen3-14B", "rough_params_B": 14.0, "kind": "dense_decoder", "adapter": "qwen3"},
    {"key": "qwen3_32b", "label": "Qwen3-32B", "model_path": "Qwen/Qwen3-32B", "rough_params_B": 32.0, "kind": "dense_decoder", "adapter": "qwen3"},
    {"key": "qwen3_30b_a3b", "label": "Qwen3-30B-A3B (MoE)", "model_path": "Qwen/Qwen3-30B-A3B", "rough_params_B": 30.0, "kind": "moe_decoder", "adapter": "qwen3"},
    {"key": "qwen3_235b_a22b", "label": "Qwen3-235B-A22B (MoE)", "model_path": "Qwen/Qwen3-235B-A22B", "rough_params_B": 235.0, "kind": "moe_decoder", "adapter": "qwen3"},
    # ---- qwen3_5 (3 models) ----
    {"key": "qwen3_5_27b", "label": "Qwen3.5-27B (hybrid MTP)", "model_path": "Qwen/Qwen3.5-27B", "rough_params_B": 27.0, "kind": "hybrid_decoder", "adapter": "qwen3_5"},
    {"key": "qwen3_5_35b_a3b", "label": "Qwen3.5-35B-A3B (MoE)", "model_path": "Qwen/Qwen3.5-35B-A3B", "rough_params_B": 35.0, "kind": "moe_decoder", "adapter": "qwen3_5"},
    {"key": "qwen3_5_122b_a10b", "label": "Qwen3.5-122B-A10B (MoE)", "model_path": "Qwen/Qwen3.5-122B-A10B", "rough_params_B": 122.0, "kind": "moe_decoder", "adapter": "qwen3_5"},
    # ---- qwen3_vl (3 models) ----
    {"key": "qwen3_vl_8b", "label": "Qwen3-VL-8B-Instruct", "model_path": "Qwen/Qwen3-VL-8B-Instruct", "rough_params_B": 8.0, "kind": "vlm", "adapter": "qwen3_vl"},
    {"key": "qwen3_vl_32b", "label": "Qwen3-VL-32B-Instruct", "model_path": "Qwen/Qwen3-VL-32B-Instruct", "rough_params_B": 32.0, "kind": "vlm", "adapter": "qwen3_vl"},
    {"key": "glm46v_flash_9b", "label": "GLM-4.6V-Flash 9B (VLM local)", "model_path": "/data/models/glm46v-flash-9b-glm46v-flash-9b-bf16-v1", "rough_params_B": 9.0, "kind": "vlm", "adapter": "qwen3_vl"},
]


# --- HF Hub fetch ------------------------------------------------------------

def _fetch_hf_config(repo_id, revision="main"):
    import requests
    url = (
        "https://huggingface.co/{repo}/resolve/{rev}/config.json"
        .format(repo=repo_id, rev=revision)
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    cfg = r.json()
    return {"config_json": cfg, "source": "huggingface:" + repo_id}


# --- Config normalization / YAML composition ---------------------------------

def _build_config_yaml(payload, model_cfg, model_path_hint):
    """Compose the YAML input for the simulator.

    The `task` field selects the preset:
      - full_ft  : standard distributed full-parameter fine-tune (default)
      - pretrain : from-scratch; distributed optimizer on, recompute on
      - lora     : placeholder: emits a `lora:` section but the underlying
                   simulator does not yet model LoRA. The simulation results
                   returned for `lora` mirror full-FT, so should be treated as
                   a rough upper bound for memory and a not-applicable figure
                   for throughput. See TODO in README for tracking.
    """
    task = payload.get("task") or "full_ft"
    p = payload
    hw_key = p.get("hardware") or "hygon_dcu_bw1000"
    hw = HW_PRESETS.get(hw_key, HW_PRESETS["hygon_dcu_bw1000"])

    cfg_path = None
    tmp_cfg_path = None
    if model_path_hint and Path(model_path_hint).exists():
        cfg_path = model_path_hint
    elif model_cfg is not None:
        tmp_cfg = tempfile.NamedTemporaryFile(
            prefix="hcumodel_", suffix=".json", delete=False, mode="w",
        )
        json.dump(model_cfg, tmp_cfg, indent=2)
        tmp_cfg.close()
        tmp_cfg_path = tmp_cfg.name
        cfg_path = tmp_cfg.name
    else:
        cfg_path = model_path_hint

    pp_size  = int(p.get("pp_size")  or 1)
    tp_size  = int(p.get("tp_size")  or 1)
    ep_size  = int(p.get("ep_size")  or 1)
    cp_size  = int(p.get("cp_size")  or 1)
    micro_bs = int(p.get("micro_batch_size") or 1)
    global_bs = int(p.get("global_batch_size") or 256)
    seq_len  = int(p.get("seq_length") or 4096)
    num_gpus = int(p.get("num_gpus") or max(tp_size * pp_size, 8))
    use_fp8  = bool(p.get("use_fp8", False))

    is_moe = bool(model_cfg or {}) and any(
        model_cfg.get(k) is not None
        for k in ("n_routed_experts", "num_experts", "num_local_experts")
    )



    # Task-specific knobs ----------------------------------------------
    TASK_DEFAULTS = {
        "full_ft": {"full_recompute": "false", "recompute_note": ""},
        "pretrain": {
            # Pretrain benefits from activation recompute because of long
            # sequences; turn on full_recompute to fit more tokens in HBM.
            "full_recompute": "true",
            "recompute_note":
                "# pretrain: full_recompute = true to fit longer sequences\n"
                "# distributed optimizer + ZeRO-1 stays the same\n",
        },
        "lora": {
            # LoRA mode currently reuses the full-FT simulator path
            # (hcu-train-simulator doesn't yet expose a LoRA-aware estimator).
            # The 'lora:' block emitted later is recorded for the user but not
            # interpreted; the downstream throughput/memory figures you see
            # therefore describe the *base-model* weights — interpret them as
            # the upper-bound for unfreezing, not the actual LoRA run.
            "full_recompute": "false",
            "recompute_note":
                "# lora: simulator is unaware of LoRA — numbers mirror full-FT.\n"
                "# Memory is ~base-weights only; actual LoRA memory is much lower.\n",
        },
    }
    DEFAULT_TASK = {"full_recompute": "false", "recompute_note": ""}
    # Pick task defaults with ONE .get(). A second .get(task, ...) on the
    # inner dict would always miss (the inner keys are 'full_recompute' /
    # 'recompute_note', not the task name) and silently fall back to the
    # default regardless of the selected task.
    tk = TASK_DEFAULTS.get(task, DEFAULT_TASK)
    full_recompute_str = tk["full_recompute"]
    task_note = tk.get("recompute_note", "")

    yaml_text = (
        "model_path: {cfg_path}\n"
        "parallel_config:\n"
        "  tp_size: {tp}\n"
        "  pp_size: {pp}\n"
        "  ep_size: {ep}\n"
        "  etp_size: null\n"
        "  cp_size: {cp}\n"
        "  pp_schedule: 1f1b\n"
        "  num_layers_per_vp_stage: null\n"
        "  decoder_first_pipeline_num_layers: null\n"
        "  decoder_last_pipeline_num_layers: null\n"
        "  use_distributed_optimizer: true\n"
        "  full_recompute: {full_recompute}\n"
        "  use_fp8_training: {fp8}\n"
        "  sequence_parallel: true\n"
        "  micro_batch_size: {mbs}\n"
        "  global_batch_size: {gbs}\n"
        "  seq_length: {seq}\n"
        "  num_gpus: {ng}\n"
        "  overlap_mode: auto\n"
        "  overlap_p2p_comm: false\n"
        "  ep_communication_backend: alltoall\n"
        "  tp_overlap_ratio: 0\n"
        "  dp_overlap_ratio: 0\n"
        "  ep_overlap_ratio: 0\n"
        "  etp_overlap_ratio: 0\n"
        "  edp_overlap_ratio: 0\n"
        "  pp_overlap_ratio: null\n"
        "  moe_grouped_gemm: {moe}\n"
        "  benchmark_ops: [theoretical]\n"
        "  benchmark_dtype: bf16\n"
        "  benchmark_device: cuda\n"
        "  benchmark_allow_cpu: false\n"
        "hardware_config:\n"
        "  use_bandwidth_table: true\n"
        "  fp16_tflops: {f16}\n"
        "  fp8_tflops: {f8}\n"
        "  gpus_per_node: {gpn}\n"
        "  hbm_gib: {hbm}\n"
        "  intra_bw_gbps: {intra}\n"
        "  inter_bw_gbps: {inter}\n"
        "  gemm_efficiency: {gemm_eff}\n"
        "  non_gemm_efficiency: {non_gemm_eff}\n"
        "  optimizer_efficiency: {opt_eff}\n"
        "  p2p_intra_efficiency: {p2p_intra}\n"
        "  collective_intra_efficiency: {coll_intra}\n"
        "  collective_inter_efficiency: {coll_inter}\n"
        "profile_config:\n"
        "  enabled: true\n"
        "  mode: theoretical\n"
        "  user_dir: ./profiles/user\n"
        "  update_on_benchmark: false\n"
        "  require_idle: false\n"
        "  ignore_gemm_layout: true\n"
        "  match_permuted_gemm: true\n"
        "search:\n"
        "  enabled: false\n"
        "  top_k: 10\n"
    ).format(
        cfg_path=cfg_path,
        tp=tp_size, pp=pp_size, ep=ep_size, cp=cp_size,
        fp8=str(use_fp8).lower(),
        full_recompute=full_recompute_str,
        mbs=micro_bs, gbs=global_bs, seq=seq_len, ng=num_gpus,
        moe=str(is_moe).lower(),
        f16=hw["fp16_tflops"], f8=hw["fp8_tflops"], gpn=min(int(hw.get("gpus_per_node", 8)), max(1, num_gpus)),
        hbm=hw["hbm_gib"], intra=hw["intra_bw_gbps"], inter=hw["inter_bw_gbps"],
        gemm_eff=hw["gemm_efficiency"],
        non_gemm_eff=hw["non_gemm_efficiency"],
        opt_eff=hw["optimizer_efficiency"],
        p2p_intra=hw["p2p_intra_efficiency"],
        coll_intra=hw["collective_intra_efficiency"],
        coll_inter=hw["collective_inter_efficiency"],
    )

    # LoRA placeholder block (no-op for the simulator today) ----
    lora_r       = int(payload.get("lora_r") or 16)
    lora_alpha   = int(payload.get("lora_alpha") or 32)
    lora_dropout = float(payload.get("lora_dropout") or 0.05)
    target_mods  = payload.get("lora_targets") or "q_proj,k_proj,v_proj,o_proj"
    lora_block = (
        "lora:\n"
        "  enabled: true\n"
        "  lora_r: {lora_r}\n"
        "  lora_alpha: {lora_alpha}\n"
        "  lora_dropout: {lora_dropout}\n"
        "  target_modules: {lora_targets}\n"
        "# NOTE: simulator ignores this block today.\n"
    ).format(lora_r=lora_r, lora_alpha=lora_alpha,
             lora_dropout=lora_dropout, lora_targets=target_mods)
    if task == "lora":
        yaml_text = task_note + yaml_text + lora_block
    else:
        # Prepend task comment for clarity in YAML.
        yaml_text = (task_note + yaml_text) if task_note else yaml_text

    return yaml_text, cfg_path, tmp_cfg_path


def _run_subcmd(yaml_path, subcmd, extra_args=None, timeout=60):
    """Invoke a hcu-train-sim sub-command via `python3 -m`.

    Running via `python3 cli.py` triggers an stdlib `logging` shadowing bug
    because `hcu_train_simulator/logging.py` collides with `logging`.
    `-m hcu_train_simulator.cli` is the canonical workaround and works
    identically for sub-commands like `solve` and `search`.
    """
    py = _resolve_python()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [py, "-m", "hcu_train_simulator.cli", subcmd, str(yaml_path)]
    if extra_args:
        cmd += list(extra_args)
    try:
        result = subprocess.run(
            cmd, cwd=str(REPO_DIR / "src"), env=env, capture_output=True,
            text=True, timeout=timeout,
        )
        return {"returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "",
                "stderr": "timeout after " + str(timeout) + "s"}
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": repr(e)}


def _run_train_sim(yaml_path, timeout=60):
    py = _resolve_python()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [py, str(REPO_DIR / "train_sim.py"), str(yaml_path), "--no-report"]
    try:
        result = subprocess.run(
            cmd, cwd=str(REPO_DIR), env=env, capture_output=True,
            text=True, timeout=timeout,
        )
        return {"returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "",
                "stderr": "timeout after " + str(timeout) + "s"}
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": repr(e)}


def _cap(s, n):
    s = s or ""
    if len(s) <= n:
        return s
    return s[-n:]

import re

def _table_after(text, header):
    """Find a markdown-like table that has `header` cells above it and return
    the first data row (list of stripped cells). Returns None if not found."""
    lines = text.splitlines()
    try:
        hi = next(i for i, l in enumerate(lines)
                  if all(h.strip() in l for h in header))
    except StopIteration:
        return None
    # Skip the next two lines: the +---+ divider and then data
    for j in range(hi + 1, min(hi + 5, len(lines))):
        line = lines[j]
        if line.strip().startswith('|') and '+' not in line:
            return [c.strip() for c in line.strip('|').split('|')]
    return None


def _parse_results(out):
    """Best-effort extraction of structured numbers from the simulator stdout."""
    p = {}
    text = out or ""

    sim = _table_after(text, ["compute_ms", "communication_ms", "iteration_ms"])
    if sim and len(sim) >= 5:
        p["simulation"] = {
            "compute_ms": sim[0],
            "communication_ms": sim[1],
            "iteration_ms": sim[2],
            "tgs": sim[3],
            "mfu": sim[4],
        }

    mem = _table_after(text, ["pp_rank", "wgo_gib", "activation_gib", "total_gib"])
    if mem and len(mem) >= 6:
        p["memory"] = {
            "pp_rank": mem[0],
            "wgo_gib": mem[1],
            "activation_gib": mem[2],
            "total_gib": mem[3],
            "hbm_gib": mem[4],
            "utilization": mem[5],
        }

    # Communication summary - one row per domain
    comm_header = ["domain", "raw_time_ms", "overlap_percent", "time_ms", "percent"]
    ci = None
    for i, line in enumerate(text.splitlines()):
        if all(h.strip() in line for h in comm_header):
            ci = i
            break
    if ci is not None:
        rows = []
        for j in range(ci + 2, len(text.splitlines())):
            line = text.splitlines()[j]
            if not line.strip().startswith('|') or '+' in line:
                if rows: break
                else: continue
            cells = [c.strip() for c in line.strip('|').split('|')]
            if len(cells) >= 5:
                rows.append({"domain": cells[0], "raw_time_ms": cells[1],
                             "overlap_percent": cells[2], "time_ms": cells[3],
                             "percent": cells[4]})
            if len(rows) > 8:
                break
        if rows:
            p["communication"] = rows

    # Solve parallel configurations
    sol_match = re.search(r"solve parallel configurations.*?\n(.+?)(?=\n=== |\Z)", text, re.S)
    if sol_match:
        sols = []
        block_lines = sol_match.group(1).splitlines()
        for line in block_lines:
            if 'minimum GPUs' in line or 'higher performance' in line:
                cells = [c.strip() for c in line.strip('|').split('|') if c.strip()]
                label = 'min_gpus' if 'minimum' in line else 'high_perf'
                # Header columns (18 cells):
                # 0:sol  1:gpus  2:nodes  3:tp  4:cp  5:pp  6:dp  7:ep  8:etp
                # 9:vp   10:vp_layers 11:schedule 12:ep_overlap
                # 13:first_pp 14:last_pp 15:mbs 16:gbs 17:bubble
                if len(cells) >= 18:
                    sols.append({
                        "label": label,
                        "gpus": cells[1], "nodes": cells[2],
                        "tp": cells[3], "cp": cells[4], "pp": cells[5],
                        "dp": cells[6], "ep": cells[7], "etp": cells[8],
                        "mbs": cells[15], "gbs": cells[16], "bubble": cells[17],
                    })
        if sols:
            p["solve_configs"] = sols

    # Solve performance overview
    perf_match = re.search(r"solve performance overview.*?\n(.+?)(?=\n2026|\Z)", text, re.S)
    if perf_match:
        for line in perf_match.group(1).splitlines():
            if 'minimum GPUs' in line or 'higher performance' in line:
                cells = [c.strip() for c in line.strip('|').split('|') if c.strip()]
                if len(cells) >= 4:
                    key = 'min_gpus' if 'minimum' in line else 'high_perf'
                    p.setdefault('solve_perf', {})[key] = {
                        "iter_ms": cells[1], "mfu": cells[2], "tgs": cells[3],
                    }
    return p


def _run_full_pipeline(yaml_path, also_solve=True, also_search=False):
    """Run the simulator end-to-end:  simulation + solve (+ optional search)."""
    sim = _run_train_sim(yaml_path)
    out_stdout = ("=== STAGE 1: Simulation (memory + compute + comm summary) ===\n"
                  + (sim.get("stdout") or "")
                  + ("\n[stage1 returncode=" + str(sim.get("returncode")) + "]\n"
                     if sim.get("returncode") else ""))
    out_stderr = sim.get("stderr") or ""

    stages = {"simulation": {"returncode": sim["returncode"], "ok": sim["returncode"] == 0}}

    if also_solve:
        solve = _run_subcmd(yaml_path, "solve", [], timeout=60)
        out_stdout += (
            "\n\n=== STAGE 2: Solve (best-config search) ===\n"
            + (solve.get("stdout") or "")
            + ("\n[stage2 returncode=" + str(solve.get("returncode")) + "]\n"
               if solve.get("returncode") else "")
        )
        out_stderr += solve.get("stderr") or ""
        stages["solve"] = {"returncode": solve["returncode"], "ok": solve["returncode"] == 0}

    if also_search:
        search = _run_subcmd(yaml_path, "search", [], timeout=60)
        out_stdout += (
            "\n\n=== STAGE 3: Search (top-K grid) ===\n"
            + (search.get("stdout") or "")
            + ("\n[stage3 returncode=" + str(search.get("returncode")) + "]\n"
               if search.get("returncode") else "")
        )
        out_stderr += search.get("stderr") or ""
        stages["search"] = {"returncode": search["returncode"], "ok": search["returncode"] == 0}

    rc = 0
    for s in stages.values():
        if s["returncode"]:
            rc = s["returncode"]
            break

    return {
        "returncode": rc,
        "stdout": _cap(out_stdout, 9000),
        "stderr": _cap(out_stderr, 3000),
        "stages": stages,
    }


# --- Rules-based architecture summary ----------------------------------------

def _summarize_config(model_cfg):
    if not isinstance(model_cfg, dict):
        return {"arch": "unknown", "summary": "(no config)"}

    summary = {
        "model_type": model_cfg.get("model_type", ""),
        "arch": "",
        "params_B": None,
        "hidden_size": model_cfg.get("hidden_size"),
        "num_layers": model_cfg.get("num_hidden_layers") or model_cfg.get("num_layers"),
        "num_heads": model_cfg.get("num_attention_heads"),
        "num_kv_heads": model_cfg.get("num_key_value_heads"),
        "head_dim": model_cfg.get("head_dim"),
        "intermediate": model_cfg.get("intermediate_size"),
        "vocab": model_cfg.get("vocab_size"),
        "rope_theta": model_cfg.get("rope_theta"),
    }

    mt = (summary["model_type"] or "").lower()
    text_cfg = json.dumps(model_cfg).lower()

    summary["vlm"] = bool(
        "vision_config" in model_cfg
        or "image_config" in model_cfg
        or "visual" in text_cfg
        or "vit" in text_cfg
    )

    expert_keys = ["num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts"]
    n_experts = next((model_cfg.get(k) for k in expert_keys if model_cfg.get(k) is not None), None)
    if n_experts:
        summary["moe"] = {
            "num_experts": n_experts,
            "topk": model_cfg.get("num_experts_per_tok")
                     or model_cfg.get("topk")
                     or model_cfg.get("moe_top_k"),
            "shared_experts": model_cfg.get("n_shared_experts")
                             or model_cfg.get("num_shared_experts")
                             or model_cfg.get("moe_shared_expert_count"),
        }
    else:
        summary["moe"] = None

    summary["mla"] = bool(
        model_cfg.get("kv_lora_rank") is not None
        or model_cfg.get("q_lora_rank") is not None
    )

    if summary["num_heads"] and summary["num_kv_heads"]:
        nh, nkv = summary["num_heads"], summary["num_kv_heads"]
        if nkv == nh:
            summary["attention_type"] = "MHA"
        elif nkv == 1:
            summary["attention_type"] = "MQA"
        else:
            summary["attention_type"] = "GQA"
    else:
        summary["attention_type"] = None

    if mt.startswith("qwen3_5") or mt.startswith("qwen3.5"):
        summary["arch"] = "qwen3.5_hybrid"
    elif mt.startswith("qwen3"):
        summary["arch"] = "qwen3_moe" if summary["moe"] else "qwen3_dense"
    elif mt.startswith("qwen2"):
        summary["arch"] = "qwen2_vl" if summary["vlm"] else "qwen2"
    elif mt.startswith("qwen"):
        summary["arch"] = "qwen_vl" if summary["vlm"] else "qwen"
    elif mt.startswith("glm"):
        summary["arch"] = "glm_vl" if summary["vlm"] else "glm"
    elif mt.startswith("llama"):
        summary["arch"] = "llama"
    elif mt.startswith("deepseek"):
        summary["arch"] = "deepseek"
    else:
        summary["arch"] = mt or "unknown"

    try:
        h = summary["hidden_size"] or 0
        L = summary["num_layers"] or 0
        i = summary["intermediate"] or 0
        v = summary["vocab"] or 0
        per_layer = 4 * h * h + 8 * h * i
        total = L * per_layer + v * h * 2
        summary["params_B"] = round(total / 1e9, 2)
    except Exception:
        pass

    return summary


# --- Flask app ---------------------------------------------------------------

app = Flask(
    __name__,
    static_folder=str(HERE / "static"),
    static_url_path="/static",
)


@app.route("/")
def index():
    return send_from_directory(HERE / "templates", "index.html")


@app.route("/api/health")
def health():
    py = _resolve_python()
    return jsonify({
        "ok": True,
        "python": py,
        "python_version": _py_version(py),
        "simulator_dir": str(REPO_DIR),
        "simulator_exists": REPO_DIR.exists(),
        "ts": time.time(),
    })


def _py_version(p):
    try:
        return subprocess.check_output([p, "-c", "import sys;print(sys.version)"], text=True).strip()
    except Exception as e:
        return "unknown: " + str(e)


@app.route("/api/presets")
def presets():
    return jsonify({
        "hardware": [{"key": k, **v} for k, v in HW_PRESETS.items()],
        "models": MODEL_PRESETS,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    input_type = payload.get("input_type", "preset")

    model_cfg = None
    model_path_hint = None
    source_label = ""
    try:
        if input_type == "upload":
            model_cfg = payload.get("config_json")
            source_label = "(uploaded config.json)"
            if not model_cfg:
                return jsonify({"error": "config_json missing"}), 400
        elif input_type == "hf_id":
            hf_id = (payload.get("hf_id") or "").strip()
            if not hf_id:
                return jsonify({"error": "hf_id missing"}), 400
            fetched = _fetch_hf_config(hf_id)
            model_cfg = fetched["config_json"]
            source_label = fetched["source"]
        elif input_type == "preset":
            preset_key = payload.get("preset")
            preset = next((m for m in MODEL_PRESETS if m["key"] == preset_key), None)
            if not preset:
                return jsonify({"error": "unknown preset"}), 400
            hf_id = preset["model_path"]
            if hf_id.startswith("/"):
                model_path_hint = hf_id
                local_cfg = Path(hf_id) / "config.json"
                if local_cfg.exists():
                    with open(local_cfg) as f:
                        model_cfg = json.load(f)
                    source_label = "(local " + str(local_cfg) + ")"
                else:
                    source_label = "(local model not found: " + hf_id + ")"
            else:
                fetched = _fetch_hf_config(hf_id)
                model_cfg = fetched["config_json"]
                source_label = fetched["source"]
    except Exception as e:
        return jsonify({"error": "fetching model config failed: " + repr(e)}), 400

    yaml_text, cfg_path, tmp_cfg_path = _build_config_yaml(payload, model_cfg, model_path_hint)
    tmp_yaml = tempfile.NamedTemporaryFile(
        prefix="hcuconf_", suffix=".yaml", delete=False, mode="w",
    )
    tmp_yaml.write(yaml_text)
    tmp_yaml.close()

    try:
        out = _run_full_pipeline(tmp_yaml.name, also_solve=True)
        py = _resolve_python()
        first_cmd = py + " " + str(REPO_DIR / "train_sim.py") + " " + tmp_yaml.name
        parsed = _parse_results(out["stdout"])
        return jsonify({
            "yaml_text": yaml_text,
            "yaml_path": tmp_yaml.name,
            "model_config_path": cfg_path,
            "summary": _summarize_config(model_cfg) if model_cfg else None,
            "source": source_label,
            "parsed": parsed,
            "simulator": {
                "command": first_cmd,
                "returncode": out["returncode"],
                "stdout": out["stdout"],
                "stderr": out["stderr"],
                "stages": out["stages"],
            },
        })
    finally:
        try:
            os.unlink(tmp_yaml.name)
        except OSError:
            pass
        if tmp_cfg_path:
            try:
                os.unlink(tmp_cfg_path)
            except OSError:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7766"))
    host = os.environ.get("HOST", "127.0.0.1")
    print("[backend] serving http://{h}:{p}/".format(h=host, p=port))
    app.run(host=host, port=port, threaded=True, debug=False)
