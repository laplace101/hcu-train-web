"""Measured-data comparison, independent of the training mode.

The estimator kernel produces planning estimates. Measured runs on real
hardware are the ground truth. This module connects the two: it finds the
measured rows that correspond to an estimate, computes the deviation, and
states how comparable the two are.

It is deliberately **mode-agnostic**. The LoRA case has its own
calibration (`services/lora_calibration.py`) because LoRA's frozen base
weights change the memory model qualitatively. But the comparison itself —
which measured row corresponds to this estimate, by how much do they
differ, and can the two be compared at all — is the same for every mode.
Full-parameter fine-tuning, GRPO and LoRA all go through here.

The measured database is read-only. This module never writes to it.

What the comparison can and cannot say:

  * **exact match** — the measured row ran the same model at the same
    sequence length, batch size and card count. The deviation is
    meaningful.
  * **nearest match** — the closest available row. The deviation is
    reported together with what differed, so it is not mistaken for an
    exact comparison.
  * **no match** — nothing comparable. Nothing is invented.

The estimate-versus-measured comparison is always presented *next to* the
estimate, never in place of it: the estimate is what the planner needs,
the measurement is what tells the planner how much to trust it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repositories import measured as measured_repo

#: How a measured row relates to the estimate.
MATCH_EXACT = "exact"
MATCH_NEAREST = "nearest"
MATCH_NONE = "none"

#: What has to line up for a comparison to be meaningful.
COMPARABLE_FIELDS = ("gpu_count", "cutoff_len", "bs", "ds_config")


#: Where the preset models live — the measured rows are matched against
#: these slugs.
CATALOGUE_DIR = Path(__file__).resolve().parents[1] / "catalog" / "models"


def _row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """The comparable numbers from one measured row."""
    tokens_per_s = row.get("tokens_per_s")
    peak_mib = row.get("peak_mem_mib")
    return {
        "tokens_per_s": tokens_per_s,
        "peak_gib": round(peak_mib / 1024, 2) if isinstance(peak_mib, (int, float)) else None,
        "gpu_count": row.get("gpu_count"),
        "seq_length": row.get("cutoff_len"),
        "micro_batch_size": row.get("bs"),
        "grad_accum": row.get("grad_accum"),
        "ds_config": row.get("ds_config"),
        "finetuning_type": row.get("finetuning_type"),
        "lora_rank": row.get("lora_rank"),
        "status": row.get("status"),
        "run_name": row.get("run_name"),
    }


def _is_lora(found: dict[str, Any]) -> bool:
    """Whether the measured row is a LoRA run."""
    finetuning = (found.get("finetuning_type") or "").strip().lower()
    if finetuning:
        return finetuning in {"lora", "qlora"}
    # A row that carries a LoRA rank is a LoRA run.
    return found.get("lora_rank") is not None


def _deviation(estimate: float | None, measured: float | None) -> float | None:
    """The percentage difference, measured against the estimate."""
    if estimate is None or measured is None or not estimate:
        return None
    return round((measured - estimate) / estimate * 100, 2)


def find_measured(
    *,
    slug: str | None,
    gpu_count: int | None = None,
    seq_length: int | None = None,
    micro_batch_size: int | None = None,
    status: str = "OK",
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The measured row that corresponds to an estimate.

    An exact match has to agree on card count, sequence length and batch
    size. If nothing matches exactly, the closest row is returned with
    `match_quality` set to `nearest`, so the caller can say what differed.

    `rows` lets a caller pass rows it has already fetched, so the query is
    not repeated.
    """
    if rows is None:
        if not slug:
            return None
        result = measured_repo.query_measured(
            slug=slug, gpu_count=gpu_count, status=status, limit=200
        ) or {}
        rows = result.get("rows") or []
    if not rows:
        return None

    def score(row: dict[str, Any]) -> tuple[int, int]:
        """How well a row matches; lower is better."""
        penalty = 0
        if seq_length is not None and row.get("cutoff_len") not in (None, seq_length):
            penalty += 1
        if (micro_batch_size is not None
                and row.get("bs") not in (None, micro_batch_size)):
            penalty += 1
        return penalty, -int(row.get("tokens_per_s") or 0)

    ordered = sorted(rows, key=score)
    best = ordered[0]
    penalties = score(best)[0]

    metrics = _row_metrics(best)
    quality = MATCH_EXACT if penalties == 0 else MATCH_NEAREST
    differences = []
    if penalties:
        if seq_length is not None and metrics["seq_length"] not in (None, seq_length):
            differences.append(
                f"序列长度 {metrics['seq_length']} ≠ 估算的 {seq_length}")
        if (micro_batch_size is not None
                and metrics["micro_batch_size"] not in (None, micro_batch_size)):
            differences.append(
                f"micro batch {metrics['micro_batch_size']} ≠ 估算的 {micro_batch_size}")

    return {
        "match_quality": quality,
        "differences": differences,
        "row": best,
        **metrics,
    }


def compare(
    estimate: dict[str, Any],
    *,
    slug: str | None,
    gpu_count: int | None = None,
    seq_length: int | None = None,
    micro_batch_size: int | None = None,
    mode: str | None = None,
    measured_is_lora: bool = False,
) -> dict[str, Any]:
    """Compare an estimate with the measured data.

    `estimate` is the summary the estimator produced — the fields
    `tgs`, `peak_gib` and `iteration_ms` are used where present.
    """
    found = find_measured(
        slug=slug,
        gpu_count=gpu_count,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
    )

    if found is None:
        return {
            "available": False,
            "matched": False,
            "reason": "没有找到可比的实测记录",
            "suggestion": (
                "上传该模型在此配置下的实测数据后，这里会给出估算与实测的偏差"
            ),
            "measured_is_lora": measured_is_lora,
            "mode": mode,
        }

    estimate_tgs = estimate.get("tgs")
    estimate_peak = estimate.get("peak_gib")
    estimate_iteration = estimate.get("iteration_ms")

    measured_tgs = found.get("tokens_per_s")
    measured_peak = found.get("peak_gib")

    # The measured row records tokens per second; the estimator reports the
    # same quantity, so the two are directly comparable.
    tgs_deviation = _deviation(estimate_tgs, measured_tgs)
    peak_deviation = _deviation(estimate_peak, measured_peak)

    notes: list[str] = []
    # The measured row says whether it was a LoRA run. When it was, the
    # estimate has to be a LoRA estimate too for the two to be comparable —
    # a full-parameter estimate against a LoRA run differs by roughly the
    # size of the base weights' optimiser state and gradients.
    row_is_lora = _is_lora(found)
    if row_is_lora and not measured_is_lora:
        notes.append(
            "实测记录是 LoRA 微调（基座权重冻结），而这里的估算来自全参训练内核，"
            "两者不是同一口径，偏差不代表估算的误差"
        )
    elif measured_is_lora:
        notes.append(
            "实测记录来自 LoRA 微调（基座权重冻结），与全参训练的估算不是同一口径"
        )
    if found["match_quality"] == MATCH_NEAREST:
        notes.append(
            "没有完全一致的实测记录，下面是配置最接近的一条：" +
            "；".join(found["differences"])
        )
    if mode:
        notes.append(f"该实测记录用于 {mode} 场景的对照")

    return {
        "available": True,
        "matched": True,
        "match_quality": found["match_quality"],
        "differences": found["differences"],
        "estimate_tgs": estimate_tgs,
        "measured_tgs": measured_tgs,
        "tgs_deviation_percent": tgs_deviation,
        "estimate_peak_gib": estimate_peak,
        "measured_peak_gib": measured_peak,
        "peak_deviation_percent": peak_deviation,
        "estimate_iteration_ms": estimate_iteration,
        "run_name": found.get("run_name"),
        "status": found.get("status"),
        "finetuning_type": found.get("finetuning_type"),
        "lora_rank": found.get("lora_rank"),
        "measured_gpu_count": found.get("gpu_count"),
        "measured_seq_length": found.get("seq_length"),
        "measured_micro_batch_size": found.get("micro_batch_size"),
        "measured_ds_config": found.get("ds_config"),
        "comparability_note": "；".join(notes),
        "measured_is_lora": row_is_lora or measured_is_lora,
        "comparable": not (row_is_lora and not measured_is_lora),
        "mode": mode,
    }


def envelope() -> dict[str, Any]:
    """Which models and configurations the measured data covers.

    The interface shows this so a user knows in advance whether a
    comparison will be possible for the model they picked.
    """
    models = measured_repo.list_measured_models() or []
    return {
        "models": models,
        "model_count": len(models),
    }
