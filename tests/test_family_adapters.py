"""Checks for the family adapters.

A family adapter must do two things:

  * **extend** — cover a model the kernel's own adapters would get wrong;
  * **not disturb** — leave the numbers of every model the kernel already
    handles bit-for-bit unchanged.

Both are checked here. The "before" numbers come from running the same
config through the kernel with the family adapters switched off, which is
what the simulator produced before this project added any family module.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_family_adapters.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine import paths  # noqa: E402

paths.bootstrap()

from engine import config_builder, families  # noqa: E402
from estimators import adapter as estimator  # noqa: E402
from services import compatibility  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


PARALLEL = {
    "tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
    "num_gpus": 8, "micro_batch_size": 1, "global_batch_size": 256,
    "seq_length": 4096,
}
WORK = PROJECT_ROOT / "runtime" / "test_family"


def _plain(value):
    """Turn the simulator's config objects into plain data."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "value") and hasattr(value, "name"):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value


def simulate(config: dict, label: str, *, with_family: bool) -> dict | None:
    """Simulate a config, with or without the family adapters."""
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{label}" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    sim_config = config_builder.build_config(
        str(path), task="full", hardware="bw1000",
        operator_profile="bw1000", parallel=dict(PARALLEL))
    sim_config = _plain(sim_config)

    previous = os.environ.get("HCU_NO_FAMILY_ADAPTERS")
    if not with_family:
        os.environ["HCU_NO_FAMILY_ADAPTERS"] = "1"
    try:
        payload = estimator._run_in_subprocess(sim_config, "simulate", timeout=300)
    except Exception as exc:  # noqa: BLE001
        print(f"      仿真失败: {type(exc).__name__}: {str(exc)[:100]}")
        return None
    finally:
        if previous is None:
            os.environ.pop("HCU_NO_FAMILY_ADAPTERS", None)
        else:
            os.environ["HCU_NO_FAMILY_ADAPTERS"] = previous

    result = estimator.summarise_simulation(payload)
    result["adapter"] = result.get("model_adapter")
    return result


def catalog(slug: str) -> dict:
    path = PROJECT_ROOT / "catalog" / "models" / slug / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def probe(slug: str) -> dict | None:
    """A captured model config.

    The checked-in fixture is preferred so the test does not depend on a
    probe run; the runtime capture is the fallback.
    """
    for path in (
        PROJECT_ROOT / "tests" / "fixtures" / "model_configs" / f"{slug}.json",
        PROJECT_ROOT / "runtime" / "probe" / slug / "config.json",
    ):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> int:
    print("=== ① 家族适配器注册 ===")
    registered = families.register_all()
    check("注册了四个家族模块", len(families.FAMILY_MODULES) == 4,
          str(len(families.FAMILY_MODULES)))

    from hcu_train_simulator.models.registry import registered_model_adapters

    names = {a.name for a in registered_model_adapters()}
    for name in ("qwen_hybrid", "deepseek_family", "glm_family",
                 "moe_field_normalised"):
        check(f"{name} 已注册", name in names)

    # Registering twice must not raise — the kernel rejects duplicates.
    families.register_all()
    check("重复注册不报错", True)

    print("\n=== ② 选择：家族适配器接管正确的模型 ===")
    cases = [
        ("Qwen3-Next-80B-A3B", "Qwen-Qwen3-Next-80B-A3B-Instruct", "qwen_hybrid"),
        ("Qwen3.5-35B-A3B", "Qwen-Qwen3.5-35B-A3B", "qwen_hybrid"),
        ("MiniMax-M2", "MiniMaxAI-MiniMax-M2", "moe_field_normalised"),
    ]
    for label, slug, expected in cases:
        config = probe(slug)
        if config is None:
            print(f"SKIP  {label} —— 没有采集到 config")
            continue
        verdict = compatibility.check_compatibility(config)
        check(f"{label} → {expected}", verdict["adapter"] == expected,
              f"实际 {verdict['adapter']}")

    print("\n=== ③ 准确��：已有模型的数字逐位不变 ===")
    # These configs the kernel already handled correctly. A family adapter
    # that changed their numbers would be a regression.
    for label, slug in [
        ("DeepSeek-V3", "deepseek-v3"),
        ("GLM-5", "glm-5"),
        ("Qwen3.5-35B-A3B", "qwen3-5-35b-a3b"),
        ("Qwen3-8B", "qwen3-8b"),
        ("Qwen3-VL-8B", "qwen3-vl-8b-instruct"),
    ]:
        config = catalog(slug)
        after = simulate(config, f"{slug}-after", with_family=True)
        before = simulate(config, f"{slug}-before", with_family=False)
        if not after or not before:
            check(f"{label} 可仿真", False)
            continue
        same = (after["iteration_ms"] == before["iteration_ms"]
                and after["peak_gib"] == before["peak_gib"]
                and after["tgs"] == before["tgs"])
        check(f"{label} 数字不变", same,
              f"{after['adapter']} {after['iteration_ms']} vs "
              f"{before['adapter']} {before['iteration_ms']}")

    print("\n=== ④ 效果：新覆盖的模型得到更准的数字 ===")
    next_config = probe("Qwen-Qwen3-Next-80B-A3B-Instruct")
    if next_config:
        after = simulate(next_config, "next-after", with_family=True)
        before = simulate(next_config, "next-before", with_family=False)
        if after and before:
            check("Qwen3-Next 落到家族适配器",
                  after["adapter"] == "qwen_hybrid", str(after["adapter"]))
            check("Qwen3-Next 之前落到 qwen3",
                  before["adapter"] == "qwen3", str(before["adapter"]))
            # The hybrid layout has to change the estimate — otherwise the
            # adapter is not doing anything.
            check("Qwen3-Next 数字随之改变",
                  after["iteration_ms"] != before["iteration_ms"],
                  f"{before['iteration_ms']} → {after['iteration_ms']}")
            check("Qwen3-Next 可估算",
                  compatibility.check_compatibility(next_config)["supported"])

    mini = probe("MiniMaxAI-MiniMax-M2")
    if mini:
        verdict = compatibility.check_compatibility(mini)
        check("MiniMax-M2 仍被拒绝", verdict["supported"] is False,
              str(verdict["reasons"])[:70])
        check("拒绝原因是未建模的结构而非字段名",
              any("无法表示" in r for r in verdict["reasons"]),
              verdict["reasons"][0][:60] if verdict["reasons"] else "")

    print("\n=== ⑤ DeepSeek DSA 系列被家族适配器接管 ===")
    v32 = probe("deepseek-ai-DeepSeek-V3.2-Exp")
    if v32:
        verdict = compatibility.check_compatibility(v32)
        check("V3.2 落到 deepseek_family",
              verdict["adapter"] == "deepseek_family", str(verdict["adapter"]))
        check("V3.2 可估算（DSA 已建模）", verdict["supported"] is True,
              str(verdict["reasons"])[:70])

    print("\n=== ⑥ 字段归一化不改变已有模型 ===")
    # The normalising adapter must not touch a config the kernel reads.
    deepseek = catalog("deepseek-v3")
    from engine.families.moe_fields import MoEFieldNormalisingAdapter

    normaliser = MoEFieldNormalisingAdapter()
    needs = normaliser._needs_normalisation(
        normaliser._language(deepseek))
    check("DeepSeek-V3 不需要归一化", needs is False)
    # None, not 0: a score of 0 would still make the adapter a candidate,
    # and the kernel's tie-break on the adapter name would then pick it
    # over the generic fallback for every dense model.
    check("DeepSeek-V3 不匹配归一化适配器",
          normaliser.match_score(deepseek) is None,
          str(normaliser.match_score(deepseek)))

    minimax = probe("MiniMaxAI-MiniMax-M2")
    if minimax:
        check("MiniMax-M2 需要归一化",
              normaliser._needs_normalisation(
                  normaliser._language(minimax)) is True)
        check("MiniMax-M2 匹配归一化适配器",
              normaliser.match_score(minimax) == 1)

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL FAMILY-ADAPTER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
