"""Training parameter validation (plan C05 / C06 / C07).

Every rule here produces a `ValidationIssue` with a human-readable reason and,
where possible, a concrete suggestion. The plan's acceptance criteria demand
that illegal TP/PP/GBS combinations are "全部被拦截并说明原因" — a bare
"invalid" is not acceptable.

Constraints implemented:

  * parallel divisibility      num_gpus % (tp × cp × pp) == 0
  * GQA head divisibility      num_key_value_heads % tp_size == 0
  * layer divisibility         num_hidden_layers % pp_size == 0
  * MoE expert placement      EP × ETP × EDP × PP == num_gpus
  * batch relation             GBS = MBS × DP × grad-accumulation
  * data volume                tokens ↔ samples ↔ steps ↔ epochs
  * checkpoint                 size and retention sanity
"""

from __future__ import annotations

from typing import Any

from domain.scenarios import ParallelPlan, TrainingScenario, ValidationIssue


def _issue(
    field: str, message: str, severity: str = "error", suggestion: str = ""
) -> ValidationIssue:
    return ValidationIssue(field=field, severity=severity, message=message, suggestion=suggestion)


def validate_parallel(
    plan: ParallelPlan,
    model: dict[str, Any] | None,
    *,
    gpus_per_node: int = 8,
) -> list[ValidationIssue]:
    """Check the parallel layout against the model shape and the card count."""
    issues: list[ValidationIssue] = []

    for name, value in (
        ("tp_size", plan.tp_size),
        ("cp_size", plan.cp_size),
        ("pp_size", plan.pp_size),
        ("ep_size", plan.ep_size),
        ("num_gpus", plan.num_gpus),
    ):
        if value < 1:
            issues.append(_issue(name, f"{name} 必须 ≥ 1，当前为 {value}"))

    if plan.num_gpus < 1:
        return issues

    divisor = plan.tp_size * plan.cp_size * plan.pp_size
    if divisor > 0 and plan.num_gpus % divisor != 0:
        issues.append(
            _issue(
                "parallel",
                f"卡数 {plan.num_gpus} 不能被 TP×CP×PP = {divisor} 整除，"
                f"DP 无法得到整数",
                suggestion="调整 TP/CP/PP 或卡数，使乘积整除卡数",
            )
        )

    if plan.num_gpus % gpus_per_node != 0:
        issues.append(
            _issue(
                "num_gpus",
                f"卡数 {plan.num_gpus} 不是每节点卡数 {gpus_per_node} 的整数倍，"
                f"无法按整节点分配",
                severity="warning",
                suggestion=f"改为 {gpus_per_node} 的整数倍（如 {plan.num_gpus // gpus_per_node * gpus_per_node}）",
            )
        )

    if model:
        kv_heads = model.get("num_key_value_heads") or model.get("num_query_groups")
        if kv_heads and plan.tp_size > 1 and kv_heads % plan.tp_size != 0:
            issues.append(
                _issue(
                    "tp_size",
                    f"KV 头数 {kv_heads} 不能被 TP={plan.tp_size} 整除，"
                    f"张量并行会切碎注意力头",
                    suggestion=f"将 TP 改为 {kv_heads} 的因数（≤{kv_heads}）",
                )
            )

        heads = model.get("num_attention_heads")
        if heads and plan.tp_size > 1 and heads % plan.tp_size != 0:
            issues.append(
                _issue(
                    "tp_size",
                    f"注意力头数 {heads} 不能被 TP={plan.tp_size} 整除",
                    suggestion=f"将 TP 改为 {heads} 的因数（≤{heads}）",
                )
            )

        layers = model.get("num_hidden_layers")
        if layers and plan.pp_size > 1 and layers % plan.pp_size != 0:
            issues.append(
                _issue(
                    "pp_size",
                    f"层数 {layers} 不能被 PP={plan.pp_size} 整除，"
                    f"各流水线阶段负载不均",
                    suggestion=f"将 PP 改为 {layers} 的因数，或显式设置首尾阶段层数",
                )
            )

        experts = model.get("n_routed_experts") or model.get("num_experts")
        if experts:
            etp = plan.etp_size if plan.etp_size else plan.tp_size
            # EDP = num_gpus / (EP × ETP × PP); the layout is only valid when
            # the expert-parallel product divides the world size exactly.
            product = plan.ep_size * etp * plan.pp_size
            if product > 0 and plan.num_gpus % product != 0:
                issues.append(
                    _issue(
                        "ep_size",
                        f"MoE 布局不合法：EP({plan.ep_size}) × ETP({etp}) × PP({plan.pp_size}) "
                        f"= {product} 不能整除卡数 {plan.num_gpus}，"
                        f"EP×ETP×EDP×PP ≠ num_gpus 恒等式不成立",
                        suggestion="调整 EP/ETP/PP，或使用「求解最优配置」由系统搜索合法组合",
                    )
                )
            if plan.ep_size > 1 and plan.ep_size % etp != 0 and etp > 1:
                issues.append(
                    _issue(
                        "ep_size",
                        f"EP={plan.ep_size} 与 ETP={etp} 不整除，专家切分会出现不均",
                        severity="warning",
                    )
                )

    return issues


def validate_batches(plan: ParallelPlan) -> list[ValidationIssue]:
    """Check the batch-size relation (plan C06)."""
    issues: list[ValidationIssue] = []

    if plan.micro_batch_size < 1:
        issues.append(_issue("micro_batch_size", "Micro Batch 必须 ≥ 1"))
    if plan.global_batch_size < 1:
        issues.append(_issue("global_batch_size", "Global Batch 必须 ≥ 1"))
    if plan.seq_length < 1:
        issues.append(_issue("seq_length", "序列长度必须 ≥ 1"))

    if plan.micro_batch_size < 1 or plan.global_batch_size < 1:
        return issues

    dp = plan.dp_size
    divisor = plan.micro_batch_size * dp
    if divisor > 0 and plan.global_batch_size % divisor != 0:
        remainder = plan.global_batch_size % divisor
        nearest = divisor * round(plan.global_batch_size / divisor)
        issues.append(
            _issue(
                "global_batch_size",
                f"Global Batch {plan.global_batch_size} 不能被 "
                f"MBS({plan.micro_batch_size}) × DP({dp}) = {divisor} 整除"
                f"（余数 {remainder}），梯度累积步数不是整数",
                suggestion=(
                    f"将 Global Batch 改为 {divisor} 的整数倍"
                    f"（最近可用值 {nearest}），或调整 MBS/DP"
                ),
            )
        )

    grad_accum = plan.grad_accumulation
    if grad_accum > 64:
        issues.append(
            _issue(
                "global_batch_size",
                f"梯度累积步数 {grad_accum} 偏大，会显著拖慢训练",
                severity="warning",
                suggestion="增大 Micro Batch 或 DP 以减少累积步数",
            )
        )

    return issues


def validate_data(scenario: TrainingScenario) -> list[ValidationIssue]:
    """Check the data-volume inputs (plan C07)."""
    issues: list[ValidationIssue] = []

    tokens = scenario.total_tokens
    samples = scenario.total_samples
    seq = scenario.parallel.seq_length

    if tokens is None and samples is None and scenario.max_steps is None and scenario.epochs is None:
        issues.append(
            _issue(
                "total_tokens",
                "未提供任何数据量信息，无法估算总训练时长",
                severity="warning",
                suggestion="填写总 Token 数、总样本数、Epoch 或最大步数之一",
            )
        )

    if tokens is not None and tokens <= 0:
        issues.append(_issue("total_tokens", "总 Token 数必须为正"))
    if samples is not None and samples <= 0:
        issues.append(_issue("total_samples", "总样本数必须为正"))

    if samples is not None and tokens is None:
        issues.append(
            _issue(
                "total_tokens",
                "只给了样本数，无法换算总 Token（需要每样本平均长度）",
                severity="warning",
                suggestion="同时填写总 Token 数，或填写数据集平均序列长度",
            )
        )

    if scenario.epochs is not None and scenario.epochs <= 0:
        issues.append(_issue("epochs", "Epoch 数必须为正"))

    return issues


def validate_checkpoint(scenario: TrainingScenario) -> list[ValidationIssue]:
    """Check checkpoint settings (plan C08)."""
    issues: list[ValidationIssue] = []

    if scenario.checkpoint_interval is not None:
        if scenario.checkpoint_interval <= 0:
            issues.append(_issue("checkpoint_interval", "保存间隔必须 ≥ 1"))
        if scenario.checkpoint_keep < 1:
            issues.append(_issue("checkpoint_keep", "保留份数必须 ≥ 1"))

    return issues


def validate_scenario(
    scenario: TrainingScenario,
    model: dict[str, Any] | None = None,
    *,
    gpus_per_node: int = 8,
) -> list[ValidationIssue]:
    """Run every rule. Returns all issues; errors block the estimate."""
    issues: list[ValidationIssue] = []
    issues += validate_parallel(scenario.parallel, model, gpus_per_node=gpus_per_node)
    issues += validate_batches(scenario.parallel)
    issues += validate_data(scenario)
    issues += validate_checkpoint(scenario)
    return issues


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(i.severity == "error" for i in issues)
