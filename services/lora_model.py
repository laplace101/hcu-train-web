"""LoRA resource model, calibrated against measured benchmarks (plan stage F).

The estimator kernel models full-parameter training. LoRA differs in two
ways that change the numbers materially:

  * **The base weights are frozen.** There are no base gradients and no base
    optimiser state — only the adapter's parameters carry gradients and
    optimiser moments. What remains resident is the bf16 base itself.
  * **The backward pass still traverses the base.** To reach an adapter
    attached to layer *i*, the gradient has to flow back through every
    layer above it, so activation memory and the input-gradient GEMM do
    *not* shrink in proportion to the trainable parameter count.

That second point is exactly why a full-parameter estimate must not be
handed back for a LoRA run: the estimate is high, but not because the
estimate is wrong — because it answers a different question.

## How the numbers are produced

**Throughput** is analytic and calibrated:

    step_time = fixed + per_sample × micro_batch_size

`fixed` is the per-iteration overhead (ZeRO-3 gather/scatter, pipeline
bubble, optimiser step). `per_sample` is the cost of one micro-batch's
forward plus input-gradient backward. Both are fitted from the measured
runs; the fit's own mean absolute error travels with the answer as a
confidence band.

**Memory** is interpolated from the measured runs, because the activation
term depends on kernel-level details (Liger's fused kernels, ZeRO's
bucketing) that an analytic proxy does not capture. Where a configuration
has no measured row, the analytic baseline — sharded bf16 base plus an
activation proxy — is used and flagged as such.

Every result carries its provenance: which measured rows it came from,
how many samples the coefficients were fitted on, and the fit's error.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from repositories import measured as measured_repo
from services import lora_calibration as calibration_module

APP_ROOT = Path(__file__).resolve().parents[1]

#: Default DeepSpeed stage for the LoRA templates.
DEFAULT_DS_STAGE = "z3"

#: Liger's fused kernels cut activation memory substantially; the measured
#: runs show between 50% and 75% depending on the model.
LIGER_ACTIVATION_SAVING = 0.55


class LoraNotSupported(Exception):
    """The configuration is outside the measured envelope."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.details = details or []


def _catalog_entry(slug: str | None) -> dict[str, Any] | None:
    if not slug:
        return None
    path = APP_ROOT / "catalog" / "catalog.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in data.get("models", []):
        if entry.get("slug") == slug:
            return entry
    return None


def _measured_group(slug: str, ds_config: str, liger: int) -> list[dict[str, Any]]:
    """Measured rows for one (model, ZeRO stage, Liger) combination."""
    if not measured_repo.available():
        return []
    conn = measured_repo._connect()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT model, gpu_count, bs, grad_accum, cutoff_len, lora_rank, ds_config, "
            "liger, status, step_time_s, tokens_per_s, tokens_per_s_total, peak_mem_mib "
            "FROM v_train_bench WHERE status='OK'"
        )]
    finally:
        conn.close()

    from services.lora_calibration import _normalise
    out = []
    for row in rows:
        if _normalise(row.get("model")) != _normalise(slug):
            continue
        if str(row.get("ds_config") or "none") != ds_config:
            continue
        if int(row.get("liger") or 0) != liger:
            continue
        out.append(row)
    return out


def _interpolate(xs: list[float], ys: list[float], x: float) -> tuple[float, str]:
    """Linear interpolation with linear extrapolation at the edges.

    Returns the value and how it was obtained, so the caller can report
    "interpolated" versus "extrapolated" honestly.
    """
    if not xs:
        raise LoraNotSupported("没有可用的实测数据点")
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]

    # A request that lands exactly on a measured point is not an
    # extrapolation, so the boundary is checked with a tolerance.
    if x < xs[0] - 1e-9:
        if len(xs) == 1:
            return ys[0], "single_point"
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return ys[0] + slope * (x - xs[0]), "extrapolated"
    if x > xs[-1] + 1e-9:
        if len(xs) == 1:
            return ys[0], "single_point"
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + slope * (x - xs[-1]), "extrapolated"
    if x <= xs[0]:
        return ys[0], "measured"
    if x >= xs[-1]:
        return ys[-1], "measured"

    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return ys[i], "interpolated"
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i]), "interpolated"
    return ys[-1], "interpolated"


def estimate_lora(
    *,
    slug: str,
    num_gpus: int = 8,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target: str = "all",
    ds_config: str = DEFAULT_DS_STAGE,
    liger: int = 1,
    grad_accum: int = 8,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate a LoRA run.

    Returns the throughput, the memory and the provenance of both. When the
    configuration falls outside the measured envelope the estimate is still
    produced (from the analytic baseline) but it is flagged, because
    refusing outright would be less useful than saying "this is an
    extrapolation".
    """
    cal = calibration or calibration_module.build_calibration()
    entry = _catalog_entry(slug)
    if entry is None:
        raise LoraNotSupported(f"未知模型：{slug}")

    params_b = entry.get("params_b") or 0.0
    layers = entry.get("num_layers") or 0
    hidden = entry.get("hidden_size") or 0
    group = _measured_group(slug, ds_config, liger)

    # ---- throughput -------------------------------------------------
    fit = calibration_module.step_time_fit(cal, slug, ds_config, liger)
    fit_source = "measured_fit"
    if fit is None:
        fit = calibration_module.nearest_model_fit(cal, params_b, ds_config, liger)
        fit_source = "nearest_model_fit" if fit else "none"

    step_time_s = None
    throughput_confidence = None
    if fit is not None:
        step_time_s = fit["fixed_s"] + fit["per_sample_s"] * micro_batch_size
        throughput_confidence = {
            "mae_percent": fit.get("mae_percent"),
            "max_error_percent": fit.get("max_error_percent"),
            "samples": fit.get("samples"),
            "fitted_slug": fit.get("slug"),
            "source": fit_source,
            "confidence": fit.get("confidence", "measured_fit"),
            "params_gap_b": fit.get("params_gap_b"),
            "measured_bs_range": fit.get("bs_range"),
        }

    tokens_per_s_total = None
    tokens_per_s = None
    if step_time_s:
        tokens_per_s_total = micro_batch_size * seq_length * num_gpus / step_time_s
        tokens_per_s = tokens_per_s_total / num_gpus

    # ---- memory -----------------------------------------------------
    memory_source = "measured_interpolation"
    peak_gib = None
    memory_confidence = None

    same_seq = [r for r in group if r.get("cutoff_len") == seq_length and r.get("peak_mem_mib")]
    usable = same_seq or [r for r in group if r.get("peak_mem_mib")]

    if usable:
        bss = [r["bs"] for r in usable]
        peaks = [r["peak_mem_mib"] / 1024 for r in usable]
        measured_gpus = usable[0].get("gpu_count")
        try:
            peak_gib, how = _interpolate(bss, peaks, micro_batch_size)
            memory_source = how if how.startswith("measured") else f"measured_{how}"
            memory_confidence = {
                "samples": len(usable),
                "measured_bs_range": [min(bss), max(bss)],
                "requested_bs": micro_batch_size,
                "seq_match": bool(same_seq),
                "measured_gpu_count": measured_gpus,
            }
            if not same_seq:
                memory_source = "measured_interpolation_other_seq"

            # The measured rows are all on the same card count. Scaling to a
            # different count changes the resident base weights under
            # ZeRO-3, so the sharded part is adjusted and the rest is kept.
            if measured_gpus and measured_gpus != num_gpus and ds_config == "z3":
                base_gib = params_b * 1e9 / 1024 ** 3
                sharded_at_measured = base_gib / measured_gpus
                sharded_at_requested = base_gib / num_gpus
                peak_gib = peak_gib - sharded_at_measured + sharded_at_requested
                memory_source += "_rescaled_cards"
                memory_confidence["card_rescaled"] = {
                    "from": measured_gpus, "to": num_gpus,
                    "base_gib_delta": round(sharded_at_requested - sharded_at_measured, 3),
                }
        except LoraNotSupported:
            peak_gib = None

    if peak_gib is None:
        # Analytic baseline: sharded bf16 base + activation proxy.
        peak_gib = _analytic_memory(
            params_b=params_b, layers=layers, hidden=hidden,
            num_gpus=num_gpus, micro_batch_size=micro_batch_size,
            seq_length=seq_length, lora_rank=lora_rank, lora_target=lora_target,
            ds_config=ds_config, liger=liger,
        )
        memory_source = "analytic_baseline"
        memory_confidence = {
            "samples": 0,
            "note": "该配置没有实测记录，显存为解析基线，未经实测校准",
        }

    # ---- adapter state ---------------------------------------------
    adapter_params = _adapter_params(
        params_b=params_b, layers=layers, hidden=hidden,
        lora_rank=lora_rank, lora_target=lora_target,
        intermediate=entry.get("intermediate_size"),
        num_heads=entry.get("num_attention_heads"),
        num_kv_heads=entry.get("num_key_value_heads"),
    )
    # bf16 adapter weights + bf16 gradients + fp32 Adam moments (m, v) and
    # an fp32 master copy: 2 + 2 + 12 = 16 bytes per adapter parameter.
    adapter_state_gib = adapter_params * 16 / 1024 ** 3

    return {
        "mode": "sft_lora",
        "slug": slug,
        "model_name": entry.get("name"),
        "params_b": params_b,
        "num_gpus": num_gpus,
        "micro_batch_size": micro_batch_size,
        "seq_length": seq_length,
        "grad_accum": grad_accum,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target": lora_target,
        "ds_config": ds_config,
        "liger": liger,

        # throughput
        "step_time_s": round(step_time_s, 4) if step_time_s else None,
        "tokens_per_s": round(tokens_per_s, 2) if tokens_per_s else None,
        "tokens_per_s_total": round(tokens_per_s_total, 2) if tokens_per_s_total else None,
        "throughput_confidence": throughput_confidence,

        # memory
        "peak_gib": round(peak_gib, 2) if peak_gib else None,
        "memory_source": memory_source,
        "memory_confidence": memory_confidence,
        "adapter_params": adapter_params,
        "adapter_state_gib": round(adapter_state_gib, 4),

        # provenance
        "measured_rows": len(group),
        "calibration_at": cal.get("generated_at"),
        "measured_source": cal.get("database"),
    }


def _adapter_params(
    *, params_b: float, layers: int, hidden: int, lora_rank: int, lora_target: str,
    intermediate: int | None = None, num_heads: int | None = None,
    num_kv_heads: int | None = None,
) -> int:
    """Number of trainable adapter parameters.

    A LoRA adapter on a `Linear(in, out)` with rank r adds `r × (in + out)`
    parameters. The target modules are the attention projections (q, k, v,
    o) and, for the "all" variant, the MLP projections (gate, up, down).

    With r=8 this comes out around 0.3% of the base model, which is the
    order of magnitude one expects from a LoRA run.
    """
    if not layers or not hidden or not lora_rank:
        return 0
    inter = intermediate or int(hidden * 2.7)
    heads = num_heads or 1
    kv_heads = num_kv_heads or heads
    head_dim = hidden // heads if heads else hidden
    kv_dim = kv_heads * head_dim

    r = lora_rank
    # q_proj (hidden -> hidden), k_proj / v_proj (hidden -> kv_dim),
    # o_proj (hidden -> hidden)
    attention = r * (hidden + hidden) + 2 * r * (hidden + kv_dim) + r * (hidden + hidden)
    if lora_target == "attn_only":
        per_layer = attention
    else:
        # gate_proj and up_proj (hidden -> inter), down_proj (inter -> hidden)
        mlp = 2 * r * (hidden + inter) + r * (inter + hidden)
        per_layer = attention + mlp
    return int(per_layer * layers)


def _analytic_memory(
    *, params_b: float, layers: int, hidden: int, num_gpus: int,
    micro_batch_size: int, seq_length: int, lora_rank: int, lora_target: str,
    ds_config: str, liger: int,
) -> float:
    """Analytic LoRA memory baseline, used where no measurement exists.

    The base weights stay resident in bf16. Under ZeRO-3 they are sharded
    across the cards; under ZeRO-2 the optimiser state is sharded but the
    base is not.
    """
    base_gib = params_b * 1e9 / 1024 ** 3
    if ds_config == "z3":
        base_gib = base_gib / num_gpus
    elif ds_config == "z2":
        base_gib = base_gib / 2

    # Activation proxy: per layer, per token, the hidden state and the
    # attention scratch. Liger's fused kernels cut the stored activations
    # substantially.
    act_per_token_gib = layers * hidden * 2 / 1024 ** 3
    act_gib = act_per_token_gib * seq_length * micro_batch_size
    if liger:
        act_gib *= (1 - LIGER_ACTIVATION_SAVING)

    adapter = _adapter_params(
        params_b=params_b, layers=layers, hidden=hidden,
        lora_rank=lora_rank, lora_target=lora_target,
    )
    adapter_gib = adapter * 16 / 1024 ** 3

    # Runtime overhead: CUDA context, communication buffers, the allocator's
    # fragmentation. Measured runs show roughly 4 GiB of this.
    runtime_gib = 4.0

    return base_gib + act_gib + adapter_gib + runtime_gib


def _measured_card_counts(calibration: dict[str, Any], slug: str) -> list[int]:
    """Card counts the calibration actually covers for a model."""
    counts: set[int] = set()
    for fit in calibration.get("step_time", {}).get("fits", []):
        if fit.get("slug") == slug:
            counts.add(int(fit.get("measured_gpu_count") or 0))
    counts.discard(0)
    return sorted(counts)


def three_plans(
    *,
    slug: str,
    num_gpus_options: list[int] | None = None,
    micro_batch_options: list[int] | None = None,
    seq_length: int = 4096,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target: str = "all",
    ds_config: str = DEFAULT_DS_STAGE,
    liger: int = 1,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The three plan types for a LoRA run.

    Same shape as the full-parameter plans, so the UI renders both the
    same way:

      * minimum — fewest cards that fit the memory budget
      * stable  — keeps headroom and avoids the worst communication setup
      * performance — the highest throughput within the given ceiling
    """
    cal = calibration or calibration_module.build_calibration()
    options = num_gpus_options or [1, 2, 4, 8]
    batch_options = micro_batch_options or [1, 2, 4, 8, 16, 32]
    hbm_gib = 64.0
    margin_gib = 5.0

    # The step-time coefficients were fitted on the measured card count.
    # Applying them to a different count would be an extrapolation that the
    # measurements do not support, so the search is restricted to the
    # counts that were actually measured.
    measured_counts = _measured_card_counts(cal, slug)
    if measured_counts:
        restricted = [g for g in options if g in measured_counts]
        if restricted:
            options = restricted

    evaluations: list[dict[str, Any]] = []
    for gpus in options:
        for mbs in batch_options:
            try:
                est = estimate_lora(
                    slug=slug, num_gpus=gpus, micro_batch_size=mbs,
                    seq_length=seq_length, lora_rank=lora_rank,
                    lora_alpha=lora_alpha, lora_target=lora_target,
                    ds_config=ds_config, liger=liger, calibration=cal,
                )
            except LoraNotSupported:
                continue
            fits = (est.get("peak_gib") or 0) <= hbm_gib - margin_gib
            evaluations.append({**est, "fits": fits})

    if not evaluations:
        raise LoraNotSupported("没有可评估的 LoRA 配置")

    feasible = [e for e in evaluations if e["fits"]] or evaluations

    minimum = min(feasible, key=lambda e: (e["num_gpus"], e["micro_batch_size"]))

    # Stable: prefer the most headroom among the cheapest card counts, and
    # avoid configurations whose memory sits right at the limit.
    cheap = [e for e in feasible if e["num_gpus"] == minimum["num_gpus"]]
    stable = max(
        cheap,
        key=lambda e: ((hbm_gib - (e["peak_gib"] or 0)) if e["peak_gib"] else 0) / hbm_gib,
    )

    performance = max(feasible, key=lambda e: e.get("tokens_per_s") or 0)

    return {
        "minimum": minimum,
        "stable": stable,
        "performance": performance,
        "evaluations": len(evaluations),
        "hbm_gib": hbm_gib,
        "margin_gib": margin_gib,
    }
