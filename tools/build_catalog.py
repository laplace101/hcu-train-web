"""Build the preset model catalog.

For every entry we:
  1. download the real HuggingFace config.json (vendored, so the app works offline),
  2. resolve which simulator model adapter the repo will route it to,
  3. derive valid TP / PP / EP options from the model's own dimensions,
  4. run the simulator once to record a verified baseline.

Step 4 is the accuracy gate: a model is only published as `verified: true` when
the recorded baseline reproduces the simulator's own summary numbers.

Usage:  python tools/build_catalog.py [--only slug] [--skip-verify]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import config_builder, engine, hf_client, net, paths  # noqa: E402

# (hf_id, family label, display name, kind)
# kind: dense | moe | vlm  -- used for grouping in the UI only.
MODELS: list[tuple[str, str, str, str]] = [
    # ---- Qwen3 dense (simulator adapter: qwen3) ----
    ("Qwen/Qwen3-0.6B", "Qwen3", "Qwen3-0.6B", "dense"),
    ("Qwen/Qwen3-1.7B", "Qwen3", "Qwen3-1.7B", "dense"),
    ("Qwen/Qwen3-4B", "Qwen3", "Qwen3-4B", "dense"),
    ("Qwen/Qwen3-8B", "Qwen3", "Qwen3-8B", "dense"),
    ("Qwen/Qwen3-14B", "Qwen3", "Qwen3-14B", "dense"),
    ("Qwen/Qwen3-32B", "Qwen3", "Qwen3-32B", "dense"),
    # ---- Qwen3 MoE ----
    ("Qwen/Qwen3-30B-A3B", "Qwen3", "Qwen3-30B-A3B", "moe"),
    ("Qwen/Qwen3-235B-A22B", "Qwen3", "Qwen3-235B-A22B", "moe"),
    # ---- Qwen3.5 dense (adapter: qwen3_5, nested text_config) ----
    ("Qwen/Qwen3.5-0.8B", "Qwen3.5", "Qwen3.5-0.8B", "dense"),
    ("Qwen/Qwen3.5-2B", "Qwen3.5", "Qwen3.5-2B", "dense"),
    ("Qwen/Qwen3.5-4B", "Qwen3.5", "Qwen3.5-4B", "dense"),
    ("Qwen/Qwen3.5-9B", "Qwen3.5", "Qwen3.5-9B", "dense"),
    ("Qwen/Qwen3.5-27B", "Qwen3.5", "Qwen3.5-27B", "dense"),
    # ---- Qwen3.5 MoE ----
    ("Qwen/Qwen3.5-35B-A3B", "Qwen3.5", "Qwen3.5-35B-A3B", "moe"),
    ("Qwen/Qwen3.5-122B-A10B", "Qwen3.5", "Qwen3.5-122B-A10B", "moe"),
    ("Qwen/Qwen3.5-397B-A17B", "Qwen3.5", "Qwen3.5-397B-A17B", "moe"),
    # ---- Qwen3-VL (adapter: qwen3_vl, nested text_config + vision_config) ----
    ("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL", "Qwen3-VL-2B", "vlm"),
    ("Qwen/Qwen3-VL-4B-Instruct", "Qwen3-VL", "Qwen3-VL-4B", "vlm"),
    ("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL", "Qwen3-VL-8B", "vlm"),
    ("Qwen/Qwen3-VL-32B-Instruct", "Qwen3-VL", "Qwen3-VL-32B", "vlm"),
    ("Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL", "Qwen3-VL-30B-A3B", "vlm"),
    ("Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL", "Qwen3-VL-235B-A22B", "vlm"),
    # ---- GLM-5 DSA (adapter: glm5) ----
    ("zai-org/GLM-5", "GLM-5", "GLM-5", "moe"),
    ("zai-org/GLM-5.2", "GLM-5", "GLM-5.2", "moe"),
    ("zai-org/GLM-5.3", "GLM-5", "GLM-5.3", "moe"),
    # ---- DeepSeek (adapter: deepseek_v3 / deepseek_v2) ----
    ("deepseek-ai/DeepSeek-V3", "DeepSeek", "DeepSeek-V3", "moe"),
    ("deepseek-ai/DeepSeek-V2", "DeepSeek", "DeepSeek-V2", "moe"),
    ("deepseek-ai/DeepSeek-V2-Lite", "DeepSeek", "DeepSeek-V2-Lite", "moe"),
    ("deepseek-ai/DeepSeek-V3.2-Exp", "DeepSeek", "DeepSeek-V3.2-Exp", "moe"),
    ("deepseek-ai/DeepSeek-R1", "DeepSeek", "DeepSeek-R1", "moe"),
    # ---- MiniMax (adapter: minimax_family) ----
    ("MiniMaxAI/MiniMax-M2", "MiniMax", "MiniMax-M2", "moe"),
    ("MiniMaxAI/MiniMax-M1", "MiniMax", "MiniMax-M1", "moe"),
    ("MiniMaxAI/MiniMax-Text-01", "MiniMax", "MiniMax-Text-01", "moe"),
    # ---- generic_gpt fallback family ----
    ("Qwen/Qwen2.5-7B", "通用 Dense", "Qwen2.5-7B", "dense"),
    ("Qwen/Qwen2.5-72B", "通用 Dense", "Qwen2.5-72B", "dense"),
    ("Qwen/Qwen2-0.5B", "通用 Dense", "Qwen2-0.5B", "dense"),
    ("mistralai/Mistral-7B-v0.3", "通用 Dense", "Mistral-7B-v0.3", "dense"),
    ("microsoft/Phi-3-mini-4k-instruct", "通用 Dense", "Phi-3-mini-4k", "dense"),
]

# Baseline verification scenario, applied to every preset.
BASELINE = {
    "seq_length": 4096,
    "micro_batch_size": 1,
    "global_batch_size": 256,
    "num_gpus": 8,
    "task": "full",
    "operator_profile": "bw1000",
}


def slug_for(hf_id: str) -> str:
    return hf_id.split("/")[-1].lower().replace(".", "-").replace("_", "-")


def text_config_of(raw: dict) -> dict:
    nested = raw.get("text_config")
    return dict(nested) if isinstance(nested, dict) else dict(raw)


def divisors_upto(value: int, limit: int = 64) -> list[int]:
    if value <= 0:
        return []
    return [d for d in range(1, min(limit, value) + 1) if value % d == 0]


def normalised_config(raw: dict) -> dict:
    """The config as the kernel's adapter would read it.

    The family adapters rename the fields the kernel looks for — the
    expert count, the expert FFN width, the attention layout — and the
    catalog has to be derived from the same view, or it would describe a
    different model from the one the simulator estimates.
    """
    try:
        from engine import families

        families.register_all()
        from hcu_train_simulator.models.registry import resolve_model_adapter

        adapter = resolve_model_adapter(raw)
        return adapter.normalize_config(raw).language
    except Exception:  # noqa: BLE001
        return text_config_of(raw)


def derive_options(raw: dict) -> dict:
    """Legal TP / PP / EP values for this model, from validate_config's asserts.

    The values come from the *normalised* config, not the raw one: several
    families write their expert count under a name the kernel does not read
    (`num_local_experts` for MiniMax, for instance), so reading the raw
    file would report a MoE model as dense.
    """
    text = normalised_config(raw)
    heads = int(text.get("num_attention_heads") or 0)
    kv_groups = int(text.get("num_key_value_heads") or heads or 0)
    hidden = int(text.get("hidden_size") or 0)
    layers = int(text.get("num_hidden_layers") or 0)
    experts = int(text.get("num_experts") or text.get("n_routed_experts") or 0)

    # hidden_size % tp, num_attention_heads % tp, num_query_groups % tp
    valid_tp = [t for t in (1, 2, 4, 8) if t <= heads and heads % t == 0 and kv_groups % t == 0]
    if hidden:
        valid_tp = [t for t in valid_tp if hidden % t == 0]
    # MLA models replace num_query_groups with projection dims; keep tp=1 safe.
    if not valid_tp:
        valid_tp = [1]

    # num_hidden_layers % pp (when no custom first/last counts)
    valid_pp = [p for p in (1, 2, 4, 8) if layers and layers % p == 0]
    if not valid_pp:
        valid_pp = [1]

    return {
        "num_layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_query_groups": kv_groups,
        "num_experts": experts or None,
        "valid_tp": valid_tp,
        "valid_pp": valid_pp,
        "valid_cp": [1, 2, 4, 8],
        # Feasibility upper bound only (widest split that keeps heads/kv
        # divisible). `_verify` overwrites this with a measured, memory-aware
        # choice; this value only survives when --skip-verify is used.
        "suggested_tp": max(valid_tp),
        "suggested_pp": valid_pp[0],
        # EP depends on the cluster size, which is only known after the first
        # estimate; `_finalize_ep` recomputes it once num_gpus is settled.
        "suggested_ep": 1,
    }


def _finalize_ep(options: dict, num_gpus: int, tp: int, pp: int) -> list[int]:
    """Legal EP values for the settled cluster shape (MoE only)."""

    if not options.get("num_experts"):
        return [1]
    return config_builder.valid_ep_options(num_gpus, tp, pp, options["num_experts"]) or [1]


def suggest_num_gpus(params: float, hbm_gib: float = 64.0, margin: float = 5.0) -> int:
    """Smallest whole-node count that fits weights+grads+optimizer states.

    Mirrors the repo's own 18-bytes-per-parameter assumption used by
    `solve_training_config` (BF16 weights + BF16 grads + FP32 Adam m/v).
    """

    usable = max(1.0, hbm_gib - margin)
    needed = params * 18 / (1024 ** 3)
    gpus = max(8, math.ceil(needed / usable))
    gpus = math.ceil(gpus / 8) * 8
    return min(gpus, 512)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="only these slugs")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    net.apply_proxy()
    paths.ensure_dirs()
    paths.bootstrap()

    # Two builds writing the catalogue at the same time would overwrite
    # each other's entries — and the slower one would win, which is how a
    # stale value survives a rebuild. The lock makes that impossible.
    lock_path = paths.APP_ROOT / "runtime" / "build_catalog.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("另一个 catalog 构建正在进行，退出。"
              "若确认没有构建在跑，删除 runtime/build_catalog.lock 后重试。")
        return 1
    try:
        os.write(lock_fd, str(os.getpid()).encode())
        os.close(lock_fd)
        return _run(args)
    finally:
        lock_path.unlink(missing_ok=True)


def _run(args) -> int:
    net.apply_proxy()
    paths.ensure_dirs()
    paths.bootstrap()

    # Register the family adapters before any config is resolved, so the
    # catalog records the adapter the simulator will actually use.
    from engine import families

    families.register_all()

    catalog: list[dict] = []
    failures: list[dict] = []

    for hf_id, family, name, kind in MODELS:
        slug = slug_for(hf_id)
        if args.only and slug not in args.only:
            continue

        model_dir = paths.MODELS_DIR / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_dir / "config.json"

        print(f"[fetch] {hf_id}", flush=True)
        try:
            if config_path.exists():
                raw = json.loads(config_path.read_text(encoding="utf-8"))
            else:
                # A config captured earlier is reused rather than fetched
                # again — the network is not always available, and the
                # captured file is the same file.
                captured = _captured_config(hf_id)
                if captured is not None:
                    raw = captured
                else:
                    raw = hf_client.fetch_hf_config(hf_id)
                config_path.write_text(
                    json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"   !! unavailable: {exc}", flush=True)
            failures.append({"hf_id": hf_id, "error": str(exc)})
            continue

        entry: dict = {
            "slug": slug,
            "name": name,
            "hf_id": hf_id,
            "family": family,
            "kind": kind,
            "config_path": str(config_path.relative_to(paths.APP_ROOT)).replace("\\", "/"),
            "model_type": raw.get("model_type"),
            "has_vision": isinstance(raw.get("vision_config"), dict),
            "nested_text_config": isinstance(raw.get("text_config"), dict),
        }
        entry.update(derive_options(raw))

        try:
            adapter = _resolve_adapter(raw)
            entry["adapter"] = adapter
        except Exception as exc:  # noqa: BLE001
            entry["adapter"] = None
            entry["adapter_error"] = str(exc)

        if not args.skip_verify:
            print(f"[verify] {hf_id}", flush=True)
            try:
                entry.update(_verify(config_path, raw))
            except Exception as exc:  # noqa: BLE001
                print(f"   !! verification failed: {exc}", flush=True)
                entry["verified"] = False
                entry["verify_error"] = str(exc)

        catalog.append(entry)
        flag = "ok " if entry.get("verified") else "-- "
        print(
            f"   {flag} adapter={entry.get('adapter')} "
            f"params={entry.get('params_b')}B tp={entry.get('suggested_tp')} "
            f"gpus={entry.get('suggested_num_gpus')}",
            flush=True,
        )

    if args.only:
        # The catalogue file holds {"models": [...]}; an older revision of
        # this script wrote a bare list, so both shapes are accepted.
        existing: list = []
        if paths.CATALOG_FILE.exists():
            raw_existing = json.loads(paths.CATALOG_FILE.read_text(encoding="utf-8"))
            if isinstance(raw_existing, dict):
                existing = raw_existing.get("models") or []
            else:
                existing = raw_existing
        merged = {e["slug"]: e for e in existing}
        merged.update({e["slug"]: e for e in catalog})
        catalog = list(merged.values())

    order = {slug_for(hf_id): i for i, (hf_id, *_rest) in enumerate(MODELS)}
    catalog.sort(key=lambda e: order.get(e["slug"], 9999))

    paths.CATALOG_FILE.write_text(
        json.dumps({"models": catalog}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    verified = sum(1 for e in catalog if e.get("verified"))
    print(f"\ncatalog written: {len(catalog)} models, {verified} verified, {len(failures)} unavailable")
    for failure in failures:
        print(f"  - {failure['hf_id']}: {failure['error']}")
    return 0


def _captured_config(hf_id: str) -> dict | None:
    """A config captured earlier, if one is available.

    The model-community and Hugging Face mirrors are not always
    reachable from the build machine, so a config that was already
    fetched is reused instead of being downloaded again.
    """
    slug = slug_for(hf_id)
    for path in (
        paths.APP_ROOT / "tests" / "fixtures" / "model_configs" / f"{slug}.json",
        paths.APP_ROOT / "runtime" / "probe" / f"{hf_id}" / "config.json",
    ):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return None


def _resolve_adapter(raw: dict) -> str:
    from hcu_train_simulator.models.registry import resolve_model_adapter

    # The family adapters have to be registered first, or the catalog would
    # record the generic fallback for models the kernel routes to a family
    # adapter.
    from engine import families

    families.register_all()

    return resolve_model_adapter(raw).name


def _run_one(config_path: Path, parallel: dict) -> dict:
    """Run the in-process engine on one config, swallowing its report tables."""
    import contextlib
    import io

    cfg = config_builder.build_config(
        str(config_path), task=BASELINE["task"], hardware="bw1000",
        operator_profile=BASELINE["operator_profile"], parallel=parallel,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.simulate(cfg, name_hint="catalog-verify")


def _probe_tp(config_path: Path, options: dict, hbm_gib: float) -> tuple[int, list[dict]]:
    """Pick TP by measured throughput inside the set that actually fits in HBM.

    The previous rule was ``max(valid_tp)`` - the widest split that keeps the
    attention heads divisible. That answers "is this legal?", not "is this
    fast?", and the two disagree badly for anything that fits on fewer cards.
    Tensor parallelism issues the same number of all-reduce / all-gather
    collectives per layer while shrinking every message, so throughput falls
    off a cliff as TP grows on small models. Measured on this engine, Qwen3-1.7B
    spends 42.7% of its iteration inside TP collectives at TP8 versus 0.3% at
    TP1, and its TGS drops from 11295 to 5329.

    So measure every legal TP, drop the ones whose peak memory exceeds the
    card, and keep the fastest survivor. When a model is too large for every
    split we keep the widest one - the UI then shows the memory it would need.
    """
    measurements: list[dict] = []
    for tp in sorted(options["valid_tp"]):
        parallel = {
            "tp_size": tp,
            "pp_size": options["suggested_pp"],
            "ep_size": 1,
            "seq_length": BASELINE["seq_length"],
            "micro_batch_size": BASELINE["micro_batch_size"],
            "global_batch_size": BASELINE["global_batch_size"],
            "num_gpus": BASELINE["num_gpus"],
        }
        try:
            outcome = _run_one(config_path, parallel)
        except Exception as exc:  # noqa: BLE001
            measurements.append({"tp_size": tp, "error": str(exc)[:160]})
            continue
        summary = outcome["summary"]
        peak = outcome["peak_memory_gib"]
        measurements.append({
            "tp_size": tp,
            "tgs": round(summary["tgs"], 2),
            "mfu": round(summary["mfu"], 5),
            "iteration_ms": round(summary["iteration_ms"], 2),
            "peak_memory_gib": round(peak, 2),
            "fits_in_hbm": peak <= hbm_gib,
        })

    fitting = [m for m in measurements if m.get("fits_in_hbm")]
    if not fitting:
        return max(options["valid_tp"]), measurements
    return max(fitting, key=lambda m: m["tgs"])["tp_size"], measurements


def _verify(config_path: Path, raw: dict) -> dict:
    options = derive_options(raw)
    hbm_gib = float(config_builder.get_hardware("bw1000")["hbm_gib"])

    # Choose TP from measurement, then probe the parameter count on one node,
    # then grow the cluster until weights + gradients + optimizer states fit.
    tp_size, tp_probe = _probe_tp(config_path, options, hbm_gib)

    parallel = {
        "tp_size": tp_size,
        "pp_size": options["suggested_pp"],
        "ep_size": 1,
        "seq_length": BASELINE["seq_length"],
        "micro_batch_size": BASELINE["micro_batch_size"],
        "global_batch_size": BASELINE["global_batch_size"],
        "num_gpus": BASELINE["num_gpus"],
    }
    outcome = _run_one(config_path, parallel)
    params = outcome["summary"]["model_params"]
    gpus = suggest_num_gpus(params)

    parallel["num_gpus"] = gpus
    ep_options = _finalize_ep(options, gpus, parallel["tp_size"], parallel["pp_size"])
    parallel["ep_size"] = ep_options[0]
    options["valid_ep"] = ep_options
    outcome = _run_one(config_path, parallel)
    summary = outcome["summary"]

    return {
        "verified": True,
        "params_b": round(summary["model_size_b"], 3),
        # Overrides the divisibility-only guess from derive_options().
        "suggested_tp": tp_size,
        "suggested_num_gpus": gpus,
        "valid_ep": ep_options,
        "suggested_ep": ep_options[0],
        "tp_probe": tp_probe,
        "baseline": {
            **BASELINE,
            "num_gpus": gpus,
            "tp_size": parallel["tp_size"],
            "pp_size": parallel["pp_size"],
            "ep_size": parallel["ep_size"],
        },
        "baseline_result": {
            "adapter": outcome["model_adapter"],
            "model_size_b": summary["model_size_b"],
            "compute_ms": summary["compute_ms"],
            "communication_ms": summary["communication_ms"],
            "iteration_ms": summary["iteration_ms"],
            "tgs": summary["tgs"],
            "mfu": summary["mfu"],
            "peak_memory_gib": outcome["peak_memory_gib"],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())