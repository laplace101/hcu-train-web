"""API access layer (plan §8).

Blueprint-based routing under `/api/v1`. The layer is deliberately thin: it
parses and validates the request shape, calls a service, and formats the
response. No computation lives here.

Error contract (plan B07 "统一配置、错误码、结构化日志"):

    {
      "ok": false,
      "error": {
        "code": "validation_failed",
        "message": "……",
        "details": [ … ]
      }
    }
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request

from domain.scenarios import (
    MODE_SPECS,
    ParallelPlan,
    TrainingMode,
    TrainingScenario,
    describe_modes,
)
from estimators.adapter import EstimationError, simulator_available, simulator_version
from repositories import images as image_repo
from repositories import measured as measured_repo
from repositories import templates as template_repo
from repositories.hardware import describe_hardware, describe_stacks, resolve_hardware
from services import compatibility
from services import estimates as estimate_service
from services import launcher
from services import measured_compare
from services import support_matrix
from services.modelscope import (
    ModelSourceError,
    fetch_modelscope_config,
    parse_uploaded_json,
)
from services.validation import has_errors, validate_scenario

api = Blueprint("api", __name__, url_prefix="/api/v1")

UPLOAD_DIR = estimate_service.APP_ROOT / "runtime" / "uploads"


def _error(code: str, message: str, details: Any = None, status: int = 400):
    body: dict[str, Any] = {"ok": False, "error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return jsonify(body), status


def _scenario_from_payload(payload: dict[str, Any]) -> TrainingScenario:
    """Build a scenario from a request body, with defaults."""
    mode_raw = payload.get("mode") or TrainingMode.SFT_FULL.value
    try:
        mode = TrainingMode(mode_raw)
    except ValueError as exc:
        raise ValueError(f"未知训练类型：{mode_raw}") from exc

    parallel_raw = payload.get("parallel") or {}
    parallel = ParallelPlan(
        tp_size=int(parallel_raw.get("tp_size", 1)),
        cp_size=int(parallel_raw.get("cp_size", 1)),
        pp_size=int(parallel_raw.get("pp_size", 1)),
        ep_size=int(parallel_raw.get("ep_size", 1)),
        etp_size=parallel_raw.get("etp_size"),
        vpp_size=int(parallel_raw.get("vpp_size", 1)),
        num_gpus=int(parallel_raw.get("num_gpus", payload.get("num_gpus", 8))),
        micro_batch_size=int(parallel_raw.get("micro_batch_size", 1)),
        global_batch_size=int(parallel_raw.get("global_batch_size", 256)),
        seq_length=int(parallel_raw.get("seq_length", 4096)),
    )

    lora_raw = payload.get("lora") or {}
    dpo_raw = payload.get("dpo") or {}
    grpo_raw = payload.get("grpo") or {}
    ppo_raw = payload.get("ppo") or {}

    return TrainingScenario(
        mode=mode,
        model_slug=payload.get("model_slug"),
        model_config_path=payload.get("model_config_path"),
        model_revision=payload.get("model_revision"),
        model_config_hash=payload.get("model_config_hash"),
        hardware_id=payload.get("hardware_id") or "bw1000",
        num_gpus=parallel.num_gpus,
        parallel=parallel,
        total_tokens=payload.get("total_tokens"),
        total_samples=payload.get("total_samples"),
        epochs=payload.get("epochs"),
        max_steps=payload.get("max_steps"),
        precision=payload.get("precision") or "bf16",
        use_distributed_optimizer=bool(payload.get("use_distributed_optimizer", True)),
        full_recompute=bool(payload.get("full_recompute", False)),
        use_fp8=bool(payload.get("use_fp8", False)),
        checkpoint_interval=payload.get("checkpoint_interval"),
        checkpoint_keep=int(payload.get("checkpoint_keep", 1)),
        save_optimizer_state=bool(payload.get("save_optimizer_state", True)),
        lora_r=int(lora_raw.get("r", 16)),
        lora_alpha=int(lora_raw.get("alpha", 32)),
        lora_dropout=float(lora_raw.get("dropout", 0.05)),
        lora_target_modules=list(lora_raw.get("target_modules") or []),
        lora_train_embeddings=bool(lora_raw.get("train_embeddings", False)),
        lora_ds_config=str(payload.get("lora_ds_config") or "z3"),
        lora_liger=int(payload.get("lora_liger", 1)),
        dpo_beta=float(dpo_raw.get("beta", 0.1)),
        dpo_preference_batch=int(dpo_raw.get("preference_batch", 1)),
        grpo_group_size=int(grpo_raw.get("group_size", 8)),
        grpo_rollout_length=int(grpo_raw.get("rollout_length", 1024)),
        grpo_kl_coef=float(grpo_raw.get("kl_coef", 0.0)),
        ppo_clip=float(ppo_raw.get("clip", 0.2)),
        ppo_epochs=int(ppo_raw.get("epochs", 1)),
        ppo_gae_lambda=float(ppo_raw.get("gae_lambda", 0.95)),
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@api.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "simulator": {
                "available": simulator_available(),
                "version": simulator_version(),
            },
            "modes": describe_modes(),
            "hardware": describe_hardware(),
        }
    )


@api.get("/training/modes")
def training_modes():
    return jsonify({"ok": True, "modes": describe_modes()})


@api.get("/hardware")
def hardware():
    return jsonify(
        {"ok": True, "hardware": describe_hardware(), "software_stacks": describe_stacks()}
    )


# ---------------------------------------------------------------------------
# Model config resolution (plan §8: POST /model-configs/resolve)
# ---------------------------------------------------------------------------


@api.post("/model-configs/resolve")
def resolve_model_config():
    """Resolve a model reference into a config.json with provenance."""
    payload = request.get_json(silent=True) or {}
    source = (payload.get("source") or "upload").lower()

    try:
        if source == "modelscope":
            resolved = fetch_modelscope_config(payload.get("reference") or "")
        elif source == "upload":
            text = payload.get("config_text")
            if not text:
                return _error("missing_config", "缺少 config.json 内容")
            resolved = parse_uploaded_json(text)
        else:
            return _error("unknown_source", f"未知的模型来源：{source}")
    except ModelSourceError as exc:
        return _error("model_source_failed", str(exc))

    facts = estimate_service.model_facts(resolved.config)
    return jsonify(
        {
            "ok": True,
            "config": resolved.config,
            "provenance": resolved.to_dict(),
            "facts": facts,
        }
    )


# ---------------------------------------------------------------------------
# Validation (plan §8: POST /training/validate)
# ---------------------------------------------------------------------------


@api.post("/training/validate")
def validate_training():
    """Validate a scenario's parameters and parallelism."""
    payload = request.get_json(silent=True) or {}
    try:
        scenario = _scenario_from_payload(payload)
    except (ValueError, TypeError) as exc:
        return _error("bad_request", str(exc))

    hardware = resolve_hardware(scenario.hardware_id)
    if hardware is None:
        return _error("unknown_hardware", f"未知硬件型号：{scenario.hardware_id}")

    facts = estimate_service.model_facts(
        estimate_service.load_model_facts(scenario.model_slug, scenario.model_config_path)
    )
    issues = validate_scenario(scenario, facts, gpus_per_node=hardware.gpus_per_node)

    spec = MODE_SPECS[scenario.mode]
    return jsonify(
        {
            "ok": not has_errors(issues),
            "issues": [i.to_dict() for i in issues],
            "mode": scenario.mode.value,
            "mode_modelled": spec.modelled,
            "mode_unmodelled_reason": spec.unmodelled_reason,
            "hardware_supported": hardware.supported,
            "hardware_reason": hardware.unsupported_reason,
        }
    )


# ---------------------------------------------------------------------------
# Estimates (plan §8: POST /training/estimates, /training/search)
# ---------------------------------------------------------------------------


@api.post("/training/estimates")
def create_estimate():
    """Simulate the scenario's current configuration."""
    return _run_estimate(action="simulate")


@api.post("/training/search")
def search_estimate():
    """Solve for the minimum / stable / performance plans."""
    return _run_estimate(action="search")


def _run_estimate(action: str):
    payload = request.get_json(silent=True) or {}
    try:
        scenario = _scenario_from_payload(payload)
    except (ValueError, TypeError) as exc:
        return _error("bad_request", str(exc))

    try:
        result = estimate_service.estimate(
            scenario, action=action, model_config=payload.get("model_config")
        )
    except estimate_service.UnsupportedModeError as exc:
        return _error("mode_not_modelled", str(exc), status=422)
    except estimate_service.ModelUnsupported as exc:
        verdict = exc.verdict
        return jsonify({
            "ok": False,
            "error": {
                "code": "model_not_supported",
                "message": "内核无法为该模型给出可信估算",
                "details": verdict.get("reasons", []),
                "suggestions": verdict.get("suggestions", []),
                "model_summary": verdict.get("model_type"),
            },
        }), 422
    except estimate_service.ScenarioRejected as exc:
        return _error("validation_failed", str(exc), getattr(exc, "issues", None))
    except EstimationError as exc:
        return _error("estimation_failed", str(exc), status=500)

    return jsonify(result)

# ---------------------------------------------------------------------------
# Images, software stacks and templates (plan §8, stage G)
# ---------------------------------------------------------------------------


@api.get("/images")
def list_images():
    """List training images from the image-hub catalogue."""
    framework = request.args.get("framework")
    keyword = request.args.get("q")
    images = image_repo.list_images(framework=framework, keyword=keyword)
    return jsonify({
        "ok": True,
        "source": image_repo.source_info(),
        "frameworks": image_repo.list_frameworks(),
        "software_stacks": image_repo.list_software_stacks(),
        "images": images,
        "count": len(images),
    })


@api.get("/images/codes")
def list_image_codes():
    """List stored extraction codes, masked (plan G02)."""
    return jsonify({"ok": True, "codes": image_repo.list_codes()})


@api.post("/images/codes")
def store_image_code():
    """Store an extraction code. The plaintext is never returned."""
    payload = request.get_json(silent=True) or {}
    repo = payload.get("repo")
    code = payload.get("code")
    if not repo or not code:
        return _error("bad_request", "需要 repo 和 code")
    result = image_repo.store_code(repo, code)
    return jsonify({"ok": True, **result})


@api.post("/images/codes/verify")
def verify_image_code():
    """Check an extraction code against the stored hash."""
    payload = request.get_json(silent=True) or {}
    repo = payload.get("repo")
    code = payload.get("code")
    if not repo or not code:
        return _error("bad_request", "需要 repo 和 code")
    return jsonify({
        "ok": True,
        "repo": repo,
        "matches": image_repo.verify_code(repo, code),
    })


@api.get("/templates")
def list_templates():
    """List training configuration templates (plan G03, G04)."""
    mode = request.args.get("mode")
    verified_only = request.args.get("verified") == "1"
    templates = template_repo.list_templates(mode, verified_only=verified_only)
    return jsonify({"ok": True, "templates": templates, "count": len(templates)})


@api.post("/launch")
def render_launch():
    """Render YAML / shell / docker commands for a template + plan (plan G05, G06)."""
    payload = request.get_json(silent=True) or {}
    template_id = payload.get("template_id")
    plan = payload.get("plan")
    if not template_id or not plan:
        return _error("bad_request", "需要 template_id 和 plan")

    template = template_repo.get_template(template_id)
    if template is None:
        return _error("unknown_template", f"未知模板：{template_id}")

    model_path = payload.get("model_path") or "YOUR_MODEL_PATH"
    output_dir = payload.get("output_dir") or "/workspace/output"
    work_dir = payload.get("work_dir") or "/workspace"
    image = payload.get("image")

    try:
        rendered = launcher.render_all(
            template_id,
            plan,
            model_path=model_path,
            output_dir=output_dir,
            work_dir=work_dir,
            image=image,
            nodes=int(payload.get("nodes", 1)),
            extra=payload.get("extra") or {},
        )
    except KeyError as exc:
        return _error("bad_request", str(exc))

    # The plan's parallelism may be nested under "parallel"; normalise it so
    # the renderer sees the same shape the estimator produced.
    return jsonify({"ok": True, **rendered})

# ---------------------------------------------------------------------------
# Measured benchmark data (read-only)
# ---------------------------------------------------------------------------


@api.get("/model-support")
def model_support():
    """The support matrix over the preset catalogue.

    Every preset model is checked against the compatibility rules, so the
    matrix reflects what the kernel can actually do.
    """
    try:
        payload = support_matrix.matrix()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": {
            "code": "support_matrix_failed",
            "message": f"生成支持矩阵失败：{exc}",
        }}), 500
    payload["ok"] = True
    return jsonify(payload)


@api.post("/model-support/check")
def model_support_check():
    """Whether one model can be estimated, and what is missing if not.

    Accepts either a resolved config, or a model reference the preset
    catalogue knows.
    """
    body = request.get_json(silent=True) or {}
    reference = str(body.get("reference") or body.get("model") or "").strip()
    config = body.get("config")

    if not reference and not config:
        return jsonify({"ok": False, "error": {
            "code": "validation_failed",
            "message": "需要提供 reference 或 config",
            "details": ["缺少模型引用"],
        }}), 400

    support = support_matrix.check_reference(
        reference,
        config=config,
        source=body.get("source"),
        revision=body.get("revision"),
    )
    # A model that cannot be estimated is still a successful query — the
    # answer is the verdict, not an error.
    return jsonify({"ok": True, **support.to_dict()})


@api.get("/model-support/upstream")
def model_support_upstream():
    """The upstream simulator's own model coverage, for comparison.

    The upstream `hcu-train-simulator` ships the same six adapters and a
    `generic_gpt` fallback, so it produces numbers for models it does not
    model. This endpoint states both behaviours side by side.
    """
    return jsonify({"ok": True, **support_matrix.upstream_baseline()})


@api.get("/model-support/adapters")
def model_support_adapters():
    """What each kernel adapter reads from a model config."""
    return jsonify({"ok": True, "adapters": support_matrix.adapter_capabilities()})


@api.get("/measured/info")
def measured_info():
    """Where the measured data comes from."""
    return jsonify({"ok": True, **measured_repo.database_info()})


@api.get("/measured/models")
def measured_models():
    """Every model present in the measured database."""
    models = measured_repo.list_measured_models()
    return jsonify({"ok": True, "models": models, "count": len(models)})


@api.get("/measured")
def measured_rows():
    """Measured rows, filterable by catalogue slug and card count."""
    slug = request.args.get("slug")
    model = request.args.get("model")
    gpu_count = request.args.get("gpu_count", type=int)
    status = request.args.get("status", "OK")
    limit = request.args.get("limit", default=50, type=int)
    result = measured_repo.query_measured(
        slug=slug, model=model, gpu_count=gpu_count,
        status=status or None, limit=limit,
    )
    return jsonify({"ok": True, **result})


@api.get("/measured/envelope")
def measured_envelope():
    """Which models and configurations the measured data covers.

    A user can check this before running an estimate, so they know in
    advance whether a comparison will be possible.
    """
    return jsonify({"ok": True, **measured_compare.envelope()})


@api.get("/measured/compare")
def measured_compare_route():
    """Compare an estimate with the measured data.

    Mode-agnostic: the same comparison serves full-parameter, LoRA and
    RL estimates. The payload states whether the two are comparable.
    """
    body = request.get_json(silent=True) or {}
    slug = body.get("slug") or body.get("model_slug")
    if not slug:
        return jsonify({"ok": False, "error": {
            "code": "validation_failed",
            "message": "需要提供 slug",
            "details": ["缺少模型标识"],
        }}), 400

    estimate = body.get("estimate") or {}
    result = measured_compare.compare(
        estimate,
        slug=slug,
        gpu_count=body.get("gpu_count"),
        seq_length=body.get("seq_length"),
        micro_batch_size=body.get("micro_batch_size"),
        mode=body.get("mode"),
        measured_is_lora=bool(body.get("measured_is_lora")),
    )
    return jsonify({"ok": True, **result})


@api.get("/measured/summary")
def measured_summary():
    """Aggregate measured rows for one catalogue slug."""
    slug = request.args.get("slug")
    if not slug:
        return _error("bad_request", "需要 slug 参数")
    summary = measured_repo.summarise_for_slug(slug)
    return jsonify({"ok": True, **summary})
