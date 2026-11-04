"""Read-only access to the local benchmark database (plan §8 "实测数据").

Source: `/root/bench_results/bench.db` — the DCU training/benchmark result
store kept on this machine. It holds LoRA fine-tuning measurements taken on
real BW1000 hardware (166 successful runs across 12 models).

**This module never writes.** The connection is opened with `mode=ro`, so the
database file cannot be modified even by accident. The plan's rule that the
product must not touch the source data is enforced at the driver level rather
than by convention.

Why the measured data matters: the estimator kernel models full-parameter
training. LoRA freezes the base weights, so it produces no base gradients and
no base optimiser state — the measured memory is roughly an order of
magnitude below the full-parameter estimate while the throughput is close.
Showing the measured rows next to the estimate is what makes that gap
visible instead of hiding it.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

#: Where the benchmark database lives. Overridable so a deployment can point
#: at its own copy.
DEFAULT_DB_PATHS = [
    Path(os.environ.get("HCU_BENCH_DB", "")) if os.environ.get("HCU_BENCH_DB") else None,
    Path("/root/bench_results/bench.db"),
    Path(__file__).resolve().parents[2] / "bench_results" / "bench.db",
]

#: Fields selected for the measured rows. `raw_json` holds the untouched
#: source row so nothing is lost when a new column is added upstream.
COLUMNS = [
    "id", "run_name", "task_desc", "host", "gpu_model", "dtk_version",
    "torch_version", "image_ref", "model", "model_dtype", "framework",
    "dataset", "max_samples", "num_train_epochs", "learning_rate",
    "lr_scheduler_type", "warmup_ratio", "gradient_checkpointing", "template",
    "finetuning_type", "gpu_count", "ds_config", "bs", "grad_accum",
    "cutoff_len", "max_steps", "liger", "flash_attn", "lora_rank",
    "lora_target", "status", "train_loss", "train_runtime_s", "step_time_s",
    "wall_s", "samples_per_s", "tokens_per_s", "tokens_per_s_total",
    "tokens_per_step", "peak_mem_mib", "mem_efficiency", "note",
]


def database_path() -> Path | None:
    for candidate in DEFAULT_DB_PATHS:
        if candidate and candidate.exists():
            return candidate
    return None


def available() -> bool:
    return database_path() is not None


def _connect() -> sqlite3.Connection:
    """Open the database read-only.

    `mode=ro` is enforced by SQLite itself: writes are rejected with
    "attempt to write a readonly database" regardless of what this process
    does.
    """
    path = database_path()
    if path is None:
        raise FileNotFoundError("未找到实测数据�� bench.db")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _normalise_model(name: str | None) -> str:
    """Normalise a model name so it can be matched against catalogue slugs.

    "Qwen/Qwen3.5-2B" and "Qwen3.5-2B" both normalise to "qwen352b".
    """
    if not name:
        return ""
    base = name.split("/")[-1]
    return (base.lower()
            .replace("-", "").replace(".", "").replace("_", ""))


def catalogue_slugs(catalogue_dir: Path) -> list[str]:
    """Slugs in the model catalogue, used to map measured models to presets."""
    if not catalogue_dir.exists():
        return []
    return sorted(p.name for p in catalogue_dir.iterdir() if p.is_dir())


def _match_slug(model: str, slugs: list[str]) -> str | None:
    """Find the catalogue slug for a measured model name."""
    target = _normalise_model(model)
    for slug in slugs:
        if _normalise_model(slug) == target:
            return slug
    return None


def query_measured(
    *,
    model: str | None = None,
    slug: str | None = None,
    gpu_count: int | None = None,
    status: str | None = "OK",
    limit: int = 50,
    catalogue_dir: Path | None = None,
) -> dict[str, Any]:
    """Return measured rows, optionally filtered.

    `slug` is the catalogue slug (`qwen3-5-2b`); `model` is the raw name as
    stored in the database (`Qwen/Qwen3.5-2B`). When only a slug is given,
    every database name that normalises to it is included.
    """
    if not available():
        return {
            "available": False,
            "reason": "未找到实测数据库 bench.db",
            "rows": [],
            "count": 0,
        }

    conn = _connect()
    try:
        where: list[str] = []
        params: list[Any] = []

        if slug:
            cat_dir = catalogue_dir or (Path(__file__).resolve().parents[1] / "catalog" / "models")
            slugs = catalogue_slugs(cat_dir)
            target = _normalise_model(slug)
            names = _names_for_slug(conn, slugs, target)
            if names:
                where.append("model IN (%s)" % ",".join("?" for _ in names))
                params.extend(names)
            else:
                where.append("model = ?")
                params.append(slug)
        elif model:
            where.append("model = ?")
            params.append(model)

        if gpu_count is not None:
            where.append("gpu_count = ?")
            params.append(gpu_count)
        if status:
            where.append("status = ?")
            params.append(status)

        sql = (
            "SELECT " + ", ".join(COLUMNS) + " FROM v_train_bench"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY tokens_per_s DESC NULLS LAST, id LIMIT ?"
        params.append(max(1, min(int(limit), 500)))

        rows = [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()

    return {
        "available": True,
        "source": str(database_path()),
        "rows": rows,
        "count": len(rows),
    }


def _names_for_slug(conn: sqlite3.Connection, slugs: list[str], target: str) -> list[str]:
    """All database model names that map to a normalised catalogue slug."""
    names = [r[0] for r in conn.execute("SELECT DISTINCT model FROM v_train_bench")]
    matched = [n for n in names if _normalise_model(n) == target]
    # Also accept the catalogue slug itself appearing verbatim.
    for slug in slugs:
        if _normalise_model(slug) == target and slug not in matched:
            matched.append(slug)
    return matched


def summarise_for_slug(slug: str, catalogue_dir: Path | None = None) -> dict[str, Any]:
    """Aggregate measured rows for one catalogue slug.

    The summary is what the UI compares against the estimate: the best
    measured throughput, the memory range, and how many runs back it.
    """
    if not available():
        return {"available": False, "slug": slug, "rows": []}

    conn = _connect()
    try:
        cat_dir = catalogue_dir or (Path(__file__).resolve().parents[1] / "catalog" / "models")
        slugs = catalogue_slugs(cat_dir)
        names = _names_for_slug(conn, slugs, _normalise_model(slug))
        if not names:
            return {"available": True, "slug": slug, "rows": [], "count": 0,
                    "reason": "实测库中没有该模型的数据"}

        placeholders = ",".join("?" for _ in names)
        rows = [dict(r) for r in conn.execute(
            f"SELECT {', '.join(COLUMNS)} FROM v_train_bench "
            f"WHERE status='OK' AND model IN ({placeholders}) "
            f"ORDER BY tokens_per_s DESC",
            names,
        )]
        oom = [dict(r) for r in conn.execute(
            f"SELECT model, gpu_count, bs, cutoff_len, lora_rank, ds_config, note "
            f"FROM v_train_bench WHERE status='OOM' AND model IN ({placeholders})",
            names,
        )]
    finally:
        conn.close()

    if not rows:
        return {"available": True, "slug": slug, "rows": [], "count": 0,
                "oom_count": len(oom), "reason": "没有成功的实测记录"}

    best = rows[0]
    peaks = [r["peak_mem_mib"] for r in rows if r.get("peak_mem_mib")]
    gpu_counts = sorted({r["gpu_count"] for r in rows if r.get("gpu_count")})

    return {
        "available": True,
        "slug": slug,
        "source": str(database_path()) if database_path() else None,
        "model_names": names,
        "count": len(rows),
        "oom_count": len(oom),
        "gpu_counts": gpu_counts,
        "best": best,
        "best_tokens_per_s": best.get("tokens_per_s"),
        "best_tokens_per_s_total": best.get("tokens_per_s_total"),
        "peak_mem_mib_min": min(peaks) if peaks else None,
        "peak_mem_mib_max": max(peaks) if peaks else None,
        "peak_mem_gib_min": round(min(peaks) / 1024, 2) if peaks else None,
        "peak_mem_gib_max": round(max(peaks) / 1024, 2) if peaks else None,
        "rows": rows,
        "oom_rows": oom[:20],
    }


def list_measured_models() -> list[dict[str, Any]]:
    """Every model in the measured database with its run counts."""
    if not available():
        return []
    conn = _connect()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT model, "
            "  COUNT(*) AS runs, "
            "  SUM(CASE WHEN status='OK' THEN 1 ELSE 0 END) AS ok_runs, "
            "  SUM(CASE WHEN status='OOM' THEN 1 ELSE 0 END) AS oom_runs, "
            "  MAX(tokens_per_s) AS best_tokens_per_s, "
            "  MIN(peak_mem_mib) AS min_peak_mem_mib "
            "FROM v_train_bench GROUP BY model ORDER BY ok_runs DESC"
        )]
    finally:
        conn.close()
    return rows


def database_info() -> dict[str, Any]:
    """Where the measured data comes from, for display next to the numbers."""
    path = database_path()
    if path is None:
        return {"available": False}
    conn = _connect()
    try:
        total = conn.execute("SELECT COUNT(*) FROM v_train_bench").fetchone()[0]
        ok = conn.execute("SELECT COUNT(*) FROM v_train_bench WHERE status='OK'").fetchone()[0]
        oom = conn.execute("SELECT COUNT(*) FROM v_train_bench WHERE status='OOM'").fetchone()[0]
        envs = [dict(r) for r in conn.execute(
            "SELECT DISTINCT gpu_model, gpu_count, dtk_version, framework FROM v_train_bench"
        )]
    finally:
        conn.close()
    return {
        "available": True,
        "path": str(path),
        "size_kib": round(path.stat().st_size / 1024, 1),
        "total_runs": total,
        "ok_runs": ok,
        "oom_runs": oom,
        "environments": envs,
        "read_only": True,
    }
