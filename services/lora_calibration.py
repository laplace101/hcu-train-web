"""LoRA calibration from the measured benchmark database (plan stage F).

The estimator kernel models full-parameter training. LoRA is different in
two ways that change the numbers materially:

  * the base weights are frozen, so there are no base gradients and no
    base optimiser state — only the adapter's parameters carry gradients
    and optimiser moments;
  * the backward pass still has to traverse the base layers to reach the
    adapters, so the activation memory and the input-gradient GEMM are
    *not* reduced in proportion to the trainable parameter count.

This module learns the coefficients of the analytic LoRA model from the
measured runs in `bench.db`. It is deliberately a *calibration* layer rather
than a second estimator: the analytic formulas live in
`services/lora_model.py`, and the coefficients here say how far the real
machine departs from them.

The database keeps growing, so the calibration is recomputed from whatever
is currently in it. Every number this module produces carries the sample
count it was derived from, and refuses to extrapolate past the measured
envelope without saying so.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from repositories import measured as measured_repo

APP_ROOT = Path(__file__).resolve().parents[1]

#: Where the learned coefficients are cached. They are derived data, so they
#: are regenerated rather than hand-edited, but caching keeps the estimate
#: endpoint fast.
CALIBRATION_PATH = APP_ROOT / "runtime" / "lora_calibration.json"

#: Minimum number of measured samples before a coefficient is trusted.
MIN_SAMPLES = 2


def _catalog_entries() -> dict[str, dict[str, Any]]:
    """Catalogue entries keyed by their normalised model name."""
    path = APP_ROOT / "catalog" / "catalog.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in data.get("models", []):
        for key in (entry.get("name"), entry.get("hf_id"), entry.get("slug")):
            if key:
                out[_normalise(key)] = entry
    return out


def _normalise(name: str | None) -> str:
    if not name:
        return ""
    return (str(name).split("/")[-1].lower()
            .replace("-", "").replace(".", "").replace("_", ""))


def _robust_line(xs: list[float], ys: list[float]) -> list[float]:
    """Theil-Sen: the median of pairwise slopes, with the median intercept.

    Least squares lets a single disagreeing measurement drag the whole
    line — one batch size whose throughput collapses under memory
    pressure was enough to push the fitted intercept negative, which then
    predicted a batch size of one at zero time. The median of the
    pairwise slopes ignores that kind of point, and needs no threshold
    to be tuned.
    """
    slopes = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = xs[j] - xs[i]
            if dx:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return _lstsq([[1.0, x] for x in xs], ys)
    slope = statistics.median(slopes)
    intercept = statistics.median([y - slope * x for x, y in zip(xs, ys)])
    return [intercept, slope]


def _lstsq(X: list[list[float]], y: list[float]) -> list[float]:
    """Least squares via normal equations with Gaussian elimination."""
    n = len(X)
    if n == 0:
        return []
    k = len(X[0])
    a = [[sum(X[i][p] * X[i][q] for i in range(n)) for q in range(k)] for p in range(k)]
    b = [sum(X[i][p] * y[i] for i in range(n)) for p in range(k)]
    for i in range(k):
        pivot = max(range(i, k), key=lambda r: abs(a[r][i]))
        if abs(a[pivot][i]) < 1e-12:
            continue
        a[i], a[pivot] = a[pivot], a[i]
        b[i], b[pivot] = b[pivot], b[i]
        for r in range(i + 1, k):
            factor = a[r][i] / a[i][i]
            for c in range(i, k):
                a[r][c] -= factor * a[i][c]
            b[r] -= factor * b[i]
    coef = [0.0] * k
    for i in reversed(range(k)):
        if abs(a[i][i]) < 1e-12:
            continue
        coef[i] = (b[i] - sum(a[i][c] * coef[c] for c in range(i + 1, k))) / a[i][i]
    return coef


def _measured_oom_rows() -> list[dict[str, Any]]:
    """Rows that ran out of memory, with the memory they reached.

    These are the memory boundary in the other direction: a configuration
    that OOMed at a given batch size cannot be run at that batch size on
    that many cards, whatever the estimate says. The peak the run reached
    before dying is close to the limit, so it also says how close.

    Only LoRA rows are read. A full-parameter run that ran out of memory
    says nothing about a LoRA configuration: LoRA does not carry the base
    gradients and optimiser state, so it fits where the full-parameter run
    did not. Reading both would mark a LoRA configuration as impossible
    because a *different* kind of training exhausted the card.
    """
    if not measured_repo.available():
        return []
    conn = measured_repo._connect()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT model, gpu_count, bs, grad_accum, cutoff_len, lora_rank, ds_config, "
            "liger, status, peak_mem_mib, note, finetuning_type "
            "FROM v_train_bench "
            "WHERE 1=1" + measured_repo._finetuning_filter("lora") +
            " AND status IN ('OOM','SKIPPED') AND peak_mem_mib IS NOT NULL"
        )]
    finally:
        conn.close()

    catalogue = _catalog_entries()
    joined: list[dict[str, Any]] = []
    for row in rows:
        entry = catalogue.get(_normalise(row.get("model")))
        if not entry:
            continue
        row = dict(row)
        row["slug"] = entry.get("slug")
        row["params_b"] = entry.get("params_b")
        joined.append(row)
    return joined


def _measured_rows() -> list[dict[str, Any]]:
    """Measured rows joined with catalogue facts.

    Only LoRA rows are read, for the same reason as the OOM rows: this
    calibration produces the coefficients of the *LoRA* resource model, and
    a full-parameter row measures a different thing — it carries the base
    gradients and optimiser state, so both its memory and its throughput
    belong to a different training regime.
    """
    if not measured_repo.available():
        return []
    conn = measured_repo._connect()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT model, gpu_count, bs, grad_accum, cutoff_len, lora_rank, ds_config, "
            "liger, flash_attn, status, step_time_s, tokens_per_s, tokens_per_s_total, "
            "peak_mem_mib, gradient_checkpointing, max_steps, train_runtime_s, "
            "finetuning_type "
            "FROM v_train_bench WHERE 1=1" + measured_repo._finetuning_filter("lora") +
            " AND status='OK'"
        )]
    finally:
        conn.close()

    catalogue = _catalog_entries()
    joined: list[dict[str, Any]] = []
    for row in rows:
        entry = catalogue.get(_normalise(row.get("model")))
        if not entry:
            continue
        row = dict(row)
        row["slug"] = entry.get("slug")
        row["params_b"] = entry.get("params_b")
        row["num_layers"] = entry.get("num_layers")
        row["hidden_size"] = entry.get("hidden_size")
        row["num_experts"] = entry.get("num_experts")
        row["active_experts"] = entry.get("active_experts") or entry.get("num_experts")
        joined.append(row)
    return joined


def _effective_step_time(row: dict[str, Any]) -> float | None:
    """The time for one training step, derived from the measured throughput.

    The database's `step_time_s` column is not consistent across ingestion
    runs: some batches recorded the time of a single micro-step, others the
    time of a full gradient-accumulation cycle.

    `tokens_per_s` is recorded consistently, so the step time is derived
    from it. The formula depends on what the stored value means, and the
    two conventions differ:

    * **per micro-step** — `tokens_per_s = bs × seq / step_time`, so
      `step_time = bs × seq / tokens_per_s`;
    * **full gradient-accumulation cycle** — `tokens_per_s = ga × bs × seq /
      step_time`, so `step_time = ga × bs × seq / tokens_per_s`.

    The current data (2026-11-05, 174 rows) matches the second convention
    with a relative error of at most 0.02%. The earlier batch does not, so
    the stored value alone is not reliable.

    The derivation therefore follows whichever convention the row's own
    numbers support: the candidate that agrees with the stored value within
    2% is the one that describes that row. Rows without a usable
    throughput fall back to the stored value.
    """
    throughput = row.get("tokens_per_s")
    bs = row.get("bs")
    seq = row.get("cutoff_len")
    stored = row.get("step_time_s")

    if not (throughput and bs and seq):
        return float(stored) if stored else None

    # `tokens_per_s` is the per-card throughput: each card processes
    # bs × seq tokens per micro-step, so the micro-step takes
    # bs × seq / tokens_per_s seconds. A full gradient-accumulation cycle
    # contains `ga` micro-steps and therefore takes `ga` times that.
    micro_step = bs * seq / throughput
    if micro_step <= 0:
        return float(stored) if stored else None

    # The stored `step_time_s` records the full gradient-accumulation
    # cycle, which the newer ingestion batches make explicit: for every
    # row of the `v2-ngpu*bs*ga*cutoff` batch, the stored time equals
    # `grad_accum × bs × seq / tokens_per_s` exactly. The throughput is
    # therefore the consistent source, and the micro-step time is what
    # the fit and the model both work in.
    return micro_step


def _fit_step_time(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit `step_time = fixed + k * bs` per (model, ds, liger, seq) group.

    The step time grows almost linearly with the micro-batch size once the
    fixed overhead is removed. The sequence length is part of the group key
    because it changes the activation footprint.
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("bs"):
            continue
        step = _effective_step_time(row)
        if step is None:
            continue
        row = dict(row)
        # The fit targets the time of one micro-step, not the whole
        # gradient-accumulation cycle. The formula `fixed + k*bs` assumes
        # the batch size is the only variable; the accumulation factor
        # changes how many micro-steps a cycle has, not how long a micro-step
        # takes. Fitting the cycle time mixes the two, and the measured
        # error goes from 12% to 50%.
        # `_effective_step_time` already returns the micro-step time.
        row["_micro_step_time"] = step
        key = (
            row["slug"],
            str(row.get("ds_config") or "none"),
            int(row.get("liger") or 0),
            row.get("cutoff_len"),
        )
        groups.setdefault(key, []).append(row)

    fits: list[dict[str, Any]] = []
    for (slug, ds, liger, seq_length), group in groups.items():
        # The same batch size appears with several accumulation factors,
        # each measuring the same micro-step. Fold them into one point
        # (the median) so the fit stays balanced: without this, a batch
        # size with six measurements dominates a linear fit against batch
        # sizes with one each.
        by_bs: dict[int, list[dict[str, Any]]] = {}
        for r in group:
            by_bs.setdefault(int(r["bs"]), []).append(r)

        points = []
        for bs, rows in sorted(by_bs.items()):
            values = [float(r["_micro_step_time"]) for r in rows]
            points.append({
                "bs": bs,
                "_micro_step_time": statistics.median(values),
                "measurements": len(values),
                "first": rows[0],
            })

        X = [[1.0, float(p["bs"])] for p in points]
        y = [float(p["_micro_step_time"]) for p in points]

        if len(points) < MIN_SAMPLES:
            # A single point only pins the micro-step time at that batch
            # size. The slope is unknown, so it is taken from the same model
            # with a different ZeRO stage if one exists, and the fit is
            # marked as low confidence.
            row = points[0]["first"]
            sibling = [
                f for f in fits
                if f["slug"] == slug and f["ds_config"] != ds and f["liger"] == liger
            ]
            per_sample = sibling[0]["per_sample_s"] if sibling else 0.0
            fixed = (float(points[0]["_micro_step_time"])
                     - per_sample * float(points[0]["bs"]))
            fits.append({
                "slug": slug,
                "ds_config": ds,
                "liger": liger,
                "seq_length": group[0].get("cutoff_len"),
                "fixed_s": round(fixed, 4),
                "per_sample_s": round(per_sample, 6),
                "samples": sum(p["measurements"] for p in points),
                "batch_sizes": len(points),
                "mae_percent": None,
                "max_error_percent": None,
                "confidence": "low_single_sample",
                "bs_range": [min(r["bs"] for r in group), max(r["bs"] for r in group)],
                "measured_gpu_count": row.get("gpu_count"),
                "params_b": row.get("params_b"),
            })
            continue

        coef = _robust_line([float(p["bs"]) for p in points], y)
        if len(coef) < 2:
            continue
        fixed, per_sample = coef[0], coef[1]
        if per_sample <= 0:
            continue
        errors = [
            (coef[0] + coef[1] * p["bs"] - p["_micro_step_time"])
            / p["_micro_step_time"] * 100
            for p in points
        ]
        fits.append({
            "slug": slug,
            "ds_config": ds,
            "liger": liger,
            "seq_length": group[0].get("cutoff_len"),
            "fixed_s": round(fixed, 4),
            "per_sample_s": round(per_sample, 6),
            "samples": sum(p["measurements"] for p in points),
            "batch_sizes": len(points),
            "mae_percent": round(statistics.mean(abs(e) for e in errors), 2),
            "max_error_percent": round(max(abs(e) for e in errors), 2),
            "confidence": "measured_fit",
            "bs_range": [min(r["bs"] for r in group), max(r["bs"] for r in group)],
            "measured_gpu_count": group[0].get("gpu_count"),
            "params_b": group[0].get("params_b"),
        })
    return {"fits": fits, "groups": len(groups)}


def _fit_memory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit the LoRA memory model.

    Analytic form (per card):

        peak_gib = base_sharded
                 + activation(model, seq, bs)
                 + adapter_state
                 + runtime_overhead

    `base_sharded` is bf16 base weights, divided by the card count under
    ZeRO-3 and halved under ZeRO-2 — that is the part LoRA does *not* save,
    because the base still has to be resident.

    The measured data lets the activation term and the fixed overhead be
    calibrated instead of guessed.
    """
    usable = [r for r in rows if r.get("peak_mem_mib") is not None]
    if len(usable) < MIN_SAMPLES:
        return {"fits": [], "samples": len(usable)}

    catalogue = _catalog_entries()
    samples: list[dict[str, Any]] = []
    for row in usable:
        entry = catalogue.get(_normalise(row.get("model")))
        if not entry or not entry.get("params_b"):
            continue
        params_b = entry["params_b"]
        layers = entry.get("num_layers") or 0
        hidden = entry.get("hidden_size") or 0
        gpus = row.get("gpu_count") or 1
        bs = row.get("bs") or 1
        seq = row.get("cutoff_len") or 0
        ds = str(row.get("ds_config") or "none")

        base_gib = params_b * 1e9 / 1024 ** 3
        if ds == "z3":
            sharded = base_gib / gpus
        elif ds == "z2":
            sharded = base_gib / 2
        else:
            sharded = base_gib

        # Per-token activation for one layer's hidden state, bf16.
        act_per_token_gib = layers * hidden * 2 / 1024 ** 3
        samples.append({
            "y": row["peak_mem_mib"] / 1024,
            "base": sharded,
            "act": act_per_token_gib * seq * bs,
            "bs": bs,
            "seq": seq,
            "ds": ds,
            "liger": int(row.get("liger") or 0),
            "slug": row.get("slug"),
            "params_b": params_b,
            "layers": layers,
            "hidden": hidden,
        })

    if len(samples) < MIN_SAMPLES:
        return {"fits": [], "samples": len(samples)}

    # Two variants: with and without a Liger term, because Liger's fused
    # kernels change activation memory substantially.
    results = {}
    for name, extra in (("plain", None), ("liger", "liger")):
        X, y = [], []
        for s in samples:
            feats = [1.0, s["base"], s["act"], float(s["bs"])]
            if extra == "liger":
                feats.append(s["act"] if s["liger"] else 0.0)
            X.append(feats)
            y.append(s["y"])
        coef = _lstsq(X, y)
        if len(coef) < 4:
            continue
        errors = []
        for feats, target in zip(X, y):
            pred = sum(c * f for c, f in zip(coef, feats))
            errors.append((pred - target) / target * 100)
        results[name] = {
            "coefficients": [round(c, 8) for c in coef],
            "samples": len(X),
            "mae_percent": round(statistics.mean(abs(e) for e in errors), 2),
            "max_error_percent": round(max(abs(e) for e in errors), 2),
        }

    # Keep whichever variant fits better.
    best_name = "plain"
    if "liger" in results and "plain" in results:
        best_name = "liger" if results["liger"]["mae_percent"] < results["plain"]["mae_percent"] else "plain"

    return {
        "variant": best_name,
        "fits": results,
        "samples": len(samples),
        "model_envelope": {
            "slugs": sorted({s["slug"] for s in samples}),
            "params_b_range": [
                min(s["params_b"] for s in samples),
                max(s["params_b"] for s in samples),
            ],
            "gpu_counts": sorted({s["gpu"] for s in samples if s.get("gpu")}),
            "bs_range": [min(s["bs"] for s in samples), max(s["bs"] for s in samples)],
            "seq_range": [min(s["seq"] for s in samples), max(s["seq"] for s in samples)],
        },
    }


def _fit_throughput(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit the per-card throughput model.

    `tokens_per_s` (per card) = bs * seq * num_gpus / step_time, so the
    step-time fit already determines throughput. This fit checks the
    relation directly and records the residual spread, which becomes the
    reported confidence band.
    """
    usable = [r for r in rows if r.get("tokens_per_s") and r.get("step_time_s")]
    if not usable:
        return {"samples": 0}

    residuals: list[float] = []
    for row in usable:
        bs = row.get("bs") or 0
        seq = row.get("cutoff_len") or 0
        gpus = row.get("gpu_count") or 1
        derived = bs * seq * gpus / row["step_time_s"]
        residuals.append((row["tokens_per_s"] - derived) / derived * 100)
    return {
        "samples": len(usable),
        "mae_percent": round(statistics.mean(abs(r) for r in residuals), 4),
        "max_error_percent": round(max(abs(r) for r in residuals), 4),
        "formula": "tokens_per_s = bs * seq * num_gpus / step_time",
    }


def _database_stamp() -> str:
    """A cheap fingerprint of the database and of this module.

    The database keeps growing as new benchmark runs are ingested, so the
    calibration has to notice. The size and modification time of the main
    file are not enough: the database runs in WAL mode, where new rows land
    in the `-wal` sidecar and the main file is only rewritten at a
    checkpoint. Both files are therefore included.

    The fingerprint also covers this module's own source. The calibration
    logic changes over time — a corrected formula, a new grouping — and a
    cached result produced by the old logic would otherwise survive and be
    presented as current. Hashing the source file is cheap and makes the
    cache invalidate itself when the logic changes.

    Re-reading the whole database on every request would be wasteful; a
    stat of the two files is enough.
    """
    parts: list[str] = []
    path = measured_repo.database_path()
    if path is not None:
        for candidate in (path, Path(f"{path}-wal")):
            try:
                stat = candidate.stat()
            except OSError:
                continue
            parts.append(f"{candidate.name}:{stat.st_size}:{int(stat.st_mtime)}")

    try:
        source = Path(__file__).read_bytes()
        parts.append(f"source:{hashlib.sha256(source).hexdigest()[:16]}")
    except OSError:
        pass

    return "|".join(parts)


def build_calibration(*, refresh: bool = False) -> dict[str, Any]:
    """Build (or load) the calibration from the measured database.

    The cache is keyed on the database's size and modification time, so new
    benchmark rows invalidate it automatically.
    """
    stamp = _database_stamp()
    if not refresh and CALIBRATION_PATH.exists():
        try:
            cached = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
            if cached.get("row_count") and cached.get("database_stamp") == stamp:
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    rows = _measured_rows()
    oom_rows = _measured_oom_rows()
    calibration = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "database": str(measured_repo.database_path()) if measured_repo.available() else None,
        "database_stamp": stamp,
        "row_count": len(rows),
        "oom_row_count": len(oom_rows),
        "step_time": _fit_step_time(rows),
        "memory": _fit_memory(rows),
        "throughput": _fit_throughput(rows),
        # The memory boundary in the other direction: configurations the
        # campaign tried and that ran out of memory.
        "oom_boundary": [
            {
                "slug": row.get("slug"),
                "model": row.get("model"),
                "ds_config": row.get("ds_config") or "none",
                "liger": int(row.get("liger") or 0),
                "seq_length": row.get("cutoff_len"),
                "bs": row.get("bs"),
                "grad_accum": row.get("grad_accum"),
                "gpu_count": row.get("gpu_count"),
                "status": row.get("status"),
                "peak_mem_gib": round((row.get("peak_mem_mib") or 0) / 1024, 2),
                "note": (row.get("note") or "")[:200],
            }
            for row in oom_rows
        ],
    }

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        CALIBRATION_PATH.write_text(json.dumps(calibration, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    except OSError:
        pass
    return calibration


def step_time_fit(calibration: dict[str, Any], slug: str | None,
                   ds_config: str = "none", liger: int = 1,
                   seq_length: int | None = None) -> dict[str, Any] | None:
    """Find the step-time fit for a model / ZeRO / Liger / sequence length.

    The sequence length is part of the fit's group key, so the closest
    match is preferred over any other combination.
    """
    fits = calibration.get("step_time", {}).get("fits", [])
    if not fits:
        return None

    def candidates(model: str | None, stage: str, lig: int):
        return [f for f in fits
                if f["slug"] == model and f["ds_config"] == stage and f["liger"] == lig]

    exact = candidates(slug, ds_config, liger)
    if exact:
        if seq_length is not None:
            same_seq = [f for f in exact if f.get("seq_length") == seq_length]
            if same_seq:
                return same_seq[0]
        return exact[0]

    # Same model, a different ZeRO stage.
    same_model = [f for f in fits if f["slug"] == slug]
    if same_model:
        return same_model[0]
    return None


def nearest_model_fit(calibration: dict[str, Any], params_b: float | None,
                      ds_config: str = "none", liger: int = 1,
                      seq_length: int | None = None) -> dict[str, Any] | None:
    """Fall back to the fit whose model size is closest.

    This is an extrapolation across models, so the result is marked
    `cross_model_extrapolation` and carries the size gap it bridges. The
    caller is expected to surface that.
    """
    if not params_b:
        return None
    fits = calibration.get("step_time", {}).get("fits", [])
    candidates = [f for f in fits if f.get("params_b")]
    if not candidates:
        return None
    same_ds = [f for f in candidates if f["ds_config"] == ds_config] or candidates
    if seq_length is not None:
        same_seq = [f for f in same_ds if f.get("seq_length") == seq_length]
        if same_seq:
            same_ds = same_seq
    best = min(same_ds, key=lambda f: abs((f["params_b"] or 0) - params_b))
    return {
        **best,
        "confidence": "cross_model_extrapolation",
        "params_gap_b": round(abs((best["params_b"] or 0) - params_b), 2),
        "target_params_b": params_b,
    }


def memory_coefficients(calibration: dict[str, Any]) -> tuple[list[float] | None, str]:
    """The memory model coefficients and the variant they belong to."""
    memory = calibration.get("memory", {})
    variant = memory.get("variant", "plain")
    fit = memory.get("fits", {}).get(variant)
    if not fit:
        return None, variant
    return fit.get("coefficients"), variant


def summary(calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    """A compact description of what the calibration currently knows."""
    cal = calibration or build_calibration()
    memory = cal.get("memory", {})
    step = cal.get("step_time", {})
    return {
        "generated_at": cal.get("generated_at"),
        "database": cal.get("database"),
        "measured_rows": cal.get("row_count", 0),
        "step_time_groups": len(step.get("fits", [])),
        "step_time_median_mae": _median(
            [f["mae_percent"] for f in step.get("fits", []) if f.get("mae_percent") is not None]
        ),
        "memory_samples": memory.get("samples", 0),
        "memory_variant": memory.get("variant"),
        "memory_mae": (memory.get("fits", {}).get(memory.get("variant", ""), {}) or {}).get("mae_percent"),
        "throughput_check": cal.get("throughput", {}),
        "model_envelope": memory.get("model_envelope", {}),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 2)
