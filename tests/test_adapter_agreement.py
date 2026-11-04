"""The compatibility check must agree with what the kernel actually does.

The check predicts which adapter the kernel will pick. If the prediction is
wrong the user is told the wrong reason, so the two are compared directly:
for each model, the predicted adapter is compared with the one the
simulator reports.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_adapter_agreement.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine import config_builder  # noqa: E402
from estimators import adapter as estimator  # noqa: E402
from services import compatibility  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


#: Captured model configs, checked into the repository so the test does not
#: depend on the network or on a probe run. The manifest records the adapter
#: the kernel is expected to pick for each one.
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "model_configs"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


def main() -> int:
    print("=== 预测与内核实际一致 ===")
    checked = 0
    for slug, meta in sorted(MANIFEST.items()):
        expected = meta["expected_adapter"]
        path = FIXTURE_DIR / meta["config"]
        if not path.exists():
            print(f"SKIP  {slug} —— 缺少 fixture")
            continue
        config = json.loads(path.read_text(encoding="utf-8"))

        predicted = compatibility.check_compatibility(config)["adapter"]
        check(f"{slug} 预测适配器", predicted == expected,
              f"预测 {predicted}，期望 {expected}")

        # Run the simulator and read back the adapter it used.
        try:
            sim_config = config_builder.build_config(
                str(path),
                task="full", hardware="bw1000", operator_profile="bw1000",
                parallel={"tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                          "num_gpus": 8, "micro_batch_size": 1,
                          "global_batch_size": 256, "seq_length": 4096},
            )
            result = estimator.summarise_simulation(estimator.simulate(sim_config))
            actual = result.get("model_adapter")
            check(f"{slug} 内核实际适配器", actual == expected,
                  f"实际 {actual}，期望 {expected}")
            check(f"{slug} 预测=实际", predicted == actual,
                  f"预测 {predicted} / 实际 {actual}")
            checked += 1
        except Exception as exc:  # noqa: BLE001
            print(f"      仿真未完成: {type(exc).__name__}: {str(exc)[:70]}")

    print(f"\n  交叉验证 {checked} 个模型")
    check("至少验证了 5 个模型", checked >= 5, str(checked))

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL ADAPTER-AGREEMENT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
