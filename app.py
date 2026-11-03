"""hcu-train-web — 海光 DCU 训练资源规划平台.

The application is layered (plan §5.1):

    api/            接入层 —— 只做解析、校验、格式化，不含计算
    services/       业务服务层 —— 场景校验、模型来源、估算编排
    estimators/     估算适配层 —— 唯一与 hcu-train-simulator 交互的地方
    repositories/   数据访问层 —— 硬件、模型目录、镜像
    domain/         领域模型 —— 训练类型、场景、并行方案

The legacy routes (`/api/simulate`, `/api/solve`, `/api/search`) are kept:
the 35/35 accuracy baseline in `verification/` exercises them, and the plan
requires that existing full-parameter / pretraining results do not regress.

New product APIs live under `/api/v1` (plan §8).
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory

# The layered packages import each other by absolute name (`from domain...`),
# so the project root has to be importable.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import catalog as catalog_module
from engine import config_builder, engine, hf_client, net, paths

from api.routes import api as api_v1

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_AS_ASCII"] = False

app.register_blueprint(api_v1)

REPORT_DIR = paths.RUNTIME_DIR / "reports"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reports/<path:name>")
def report_file(name: str):
    return send_from_directory(REPORT_DIR, name)


# ---------------------------------------------------------------------------
# Bootstrap: everything the UI needs to render its form
# ---------------------------------------------------------------------------

@app.route("/api/bootstrap")
def api_bootstrap():
    catalog = catalog_module.load_catalog()
    return jsonify(
        {
            "ok": True,
            "tasks": config_builder.TASKS,
            "hardware": list(config_builder.HARDWARE_PRESETS.values()),
            "operator_profiles": config_builder.OPERATOR_PROFILES,
            "models": catalog,
            "families": catalog_module.family_summary(catalog),
            "proxy": net.apply_proxy() or "",
            "simulator": _simulator_info(),
        }
    )


def _simulator_info() -> dict[str, Any]:
    src = paths.simulator_src()
    return {
        "path": str(src),
        "repo": str(paths.WORKSPACE_ROOT / "hcu-train-simulator"),
        "version": _package_version(),
    }


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("hcu-train-simulator")
    except Exception:  # noqa: BLE001 - not pip-installed is fine
        return "source-checkout"


# ---------------------------------------------------------------------------
# Model source resolution (HF link / upload / paste)
# ---------------------------------------------------------------------------

@app.route("/api/resolve-model", methods=["POST"])
def api_resolve_model():
    payload = request.get_json(silent=True) or {}
    # A multipart upload carries `source` as a form field, not a JSON body key.
    if not payload and request.form:
        payload = {key: request.form.get(key) for key in request.form}
    net.apply_proxy()
    try:
        raw, stored_path, label = _extract_model_payload(payload)
        summary = catalog_module.inspect_config(raw)
        return jsonify(
            {
                "ok": True,
                "summary": summary,
                "model_ref": f"path:{stored_path}",
                "label": label,
                "local_path": stored_path,
            }
        )
    except hf_client.ModelSourceError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 400


def _extract_model_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    """Return (config dict, stored config.json path, display label)."""

    source = payload.get("source")
    if source == "hf":
        raw = hf_client.fetch_hf_config(payload.get("hf_ref", ""))
        repo, _revision = hf_client.parse_hf_reference(payload.get("hf_ref", ""))
        slug = repo.split("/")[-1]
        label = repo
    elif source == "upload":
        files = request.files.getlist("file")
        if not files:
            raise hf_client.ModelSourceError("请选择 config.json 文件")
        upload = files[0]
        raw = hf_client.parse_uploaded_json(upload.read().decode("utf-8-sig"))
        slug = Path(upload.filename or "uploaded-model").stem
        label = slug
    elif source == "paste":
        raw = hf_client.parse_uploaded_json(payload.get("config_text", ""))
        slug = payload.get("slug") or "pasted-model"
        label = slug
    else:
        raise hf_client.ModelSourceError(f"未知的模型来源: {source!r}")

    path = hf_client.store_config(raw, slug)
    return raw, str(path), label


# ---------------------------------------------------------------------------
# Core estimation endpoints
# ---------------------------------------------------------------------------

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        spec = _build_from_request()
        started = time.perf_counter()
        with_report = bool((request.get_json(silent=True) or {}).get("with_report"))
        outcome = engine.simulate(spec["config"], name_hint="simulate")
        outcome["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        outcome["model_source"] = spec["source_info"]
        return jsonify({"ok": True, "result": outcome})
    except engine.SimulationError as exc:
        return jsonify({"ok": False, "error": str(exc), "log": exc.log}), 400
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/api/solve", methods=["POST"])
def api_solve():
    try:
        spec = _build_from_request()
        payload = request.get_json(silent=True) or {}
        options = {
            "max_bubble_ratio": float(payload.get("max_bubble_ratio", 0.15)),
            "memory_margin_gib": payload.get("memory_margin_gib", 5),
        }
        started = time.perf_counter()
        outcome = engine.solve(spec["config"], options=options)
        outcome["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return jsonify({"ok": True, "result": outcome})
    except engine.SimulationError as exc:
        return jsonify({"ok": False, "error": str(exc), "log": exc.log}), 400
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/api/search", methods=["POST"])
def api_search():
    try:
        spec = _build_from_request()
        payload = request.get_json(silent=True) or {}
        started = time.perf_counter()
        outcome = engine.search(spec["config"], top_k=int(payload.get("top_k", 10)))
        outcome["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return jsonify({"ok": True, "result": outcome})
    except engine.SimulationError as exc:
        return jsonify({"ok": False, "error": str(exc), "log": exc.log}), 400
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/api/health")
def api_health():
    try:
        src = paths.simulator_src()
        return jsonify({"ok": True, "simulator_src": str(src)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500


def _build_from_request() -> dict[str, Any]:
    """Turn the UI form into a simulator config plus a description of the source."""

    payload = request.get_json(silent=True) or {}
    task = payload.get("task", "full")
    hardware = payload.get("hardware", "bw1000")
    operator_profile = payload.get("operator_profile", "bw1000")

    model_ref = payload.get("model")
    if not model_ref:
        raise ValueError("请选择或输入模型")

    if model_ref.startswith("path:"):
        model_path = model_ref[len("path:"):]
        source_info = {"kind": "local", "label": Path(model_path).parent.name}
    elif model_ref.startswith("upload:"):
        model_path = model_ref[len("upload:"):]
        source_info = {"kind": "upload", "label": Path(model_path).parent.name}
    else:
        entry = catalog_module.get_entry(model_ref)
        if not entry:
            raise ValueError(f"未找到预设模型: {model_ref}")
        model_path = str(paths.APP_ROOT / entry["config_path"])
        source_info = {"kind": "preset", "label": entry["name"], "adapter": entry.get("adapter")}

    parallel = {
        key: value
        for key, value in (payload.get("parallel") or {}).items()
        if value not in ("", None)
    }
    for numeric in ("tp_size", "cp_size", "pp_size", "ep_size", "etp_size",
                    "num_layers_per_vp_stage", "decoder_first_pipeline_num_layers",
                    "decoder_last_pipeline_num_layers", "micro_batch_size",
                    "global_batch_size", "seq_length", "num_gpus"):
        if numeric in parallel and parallel[numeric] == "":
            parallel[numeric] = None
    if "use_fp8_training" in (payload.get("parallel") or {}):
        parallel["use_fp8_training"] = bool(payload["parallel"]["use_fp8_training"])

    overrides = payload.get("hardware_overrides") or {}
    overrides = {k: v for k, v in overrides.items() if v not in ("", None)}

    config = config_builder.build_config(
        model_path,
        task=task,
        hardware=hardware,
        operator_profile=operator_profile,
        parallel=parallel,
        hardware_overrides=overrides,
        search={"candidates": payload.get("search_candidates") or {}},
    )
    return {"config": config, "source_info": source_info}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7766)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    paths.ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"hcu-train-web  ->  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)