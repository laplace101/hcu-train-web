"""Does the estimator actually differentiate pretraining from full SFT?

Runs the same model and parallel layout under both task defaults and
compares the numbers, so the answer is measured rather than assumed.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_task_difference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine import config_builder  # noqa: E402
from estimators import adapter  # noqa: E402

MODEL = str(PROJECT_ROOT / "catalog" / "models" / "qwen3-8b" / "config.json")


def run(task: str) -> dict:
    cfg = config_builder.build_config(
        MODEL,
        task=task,
        hardware="bw1000",
        operator_profile="bw1000",
        parallel={"tp_size": 2, "pp_size": 1, "num_gpus": 8,
                  "micro_batch_size": 1, "global_batch_size": 256, "seq_length": 4096},
    )
    return adapter.summarise_simulation(adapter.simulate(cfg))


def main() -> int:
    print("=== 同一模型/并行方案下 full 与 pretrain 的差异 ===")
    print(f"model: {MODEL}\n")

    full = run("full")
    pre = run("pretrain")

    keys = ["iteration_ms", "compute_time_ms", "communication_time_ms",
            "mfu", "tgs", "peak_gib", "wgo_gib", "activation_gib"]
    print(f"{'字段':<26} {'full':>14} {'pretrain':>14}   相同?")
    print("-" * 68)
    same = True
    for key in keys:
        a, b = full.get(key), pre.get(key)
        is_same = a == b
        same &= is_same
        print(f"{key:<26} {str(a):>14} {str(b):>14}   {'是' if is_same else '否'}")

    print()
    print("=== 结论 ===")
    if same:
        print("两种任务当前产生完全相同的估算数字。")
        print("原因：simulator 内核没有 pretrain/SFT 的分支逻辑，")
        print("      唯一差别是 config 里 full_recompute=True，")
        print("      而内核并未消费该字段（无重计算建模）。")
    else:
        print("两种任务的估算存在差异，说明内核区分了它们。")

    print()
    print("=== 任务默认值差异 ===")
    for task in ("full", "pretrain"):
        defaults = config_builder.TASK_DEFAULTS.get(task, {})
        print(f"  {task:<10} {defaults}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
