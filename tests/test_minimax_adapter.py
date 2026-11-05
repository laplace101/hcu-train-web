"""Checks for the MiniMax family adapter.

MiniMax's generations differ in an easy-to-miss way:

  * **M2 and later** use full multi-head attention across every layer.
    The released config still carries `attn_type_list`, but with the
    default value `[1] * num_hidden_layers`, and the official modelling
    code has a single attention class with no linear-attention variant.
    Treating that field as a hybrid layout would move the estimate by
    about 12%.
  * **M1 and Text-01** interleave Lightning Attention with full attention:
    70 Lightning layers and 10 full-attention layers, one of the latter
    every eighth.

The adapter also has to read MiniMax's own field names: the expert count
is `num_local_experts`, the expert FFN width is `mlp_intermediate_size`
on M2 and `intermediate_size` on M1, and the MTP module is described by
`num_mtp_modules`.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_minimax_adapter.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine import paths  # noqa: E402

paths.bootstrap()

from engine import config_builder, families  # noqa: E402
from engine.families.minimax import MiniMaxFamilyAdapter  # noqa: E402
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
WORK = PROJECT_ROOT / "runtime" / "test_minimax"
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "model_configs"


def fixture(slug: str) -> dict | None:
    """A captured MiniMax config."""
    for path in (
        FIXTURES / f"{slug}.json",
        PROJECT_ROOT / "runtime" / "probe" / slug / "config.json",
    ):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def simulate(config: dict, label: str) -> dict | None:
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{label}" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        sim_config = config_builder.build_config(
            str(path), task="full", hardware="bw1000",
            operator_profile="bw1000", parallel=dict(PARALLEL))
        result = estimator.summarise_simulation(estimator.simulate(sim_config))
    except Exception as exc:  # noqa: BLE001
        print(f"      仿真失败: {type(exc).__name__}: {str(exc)[:110]}")
        return None
    result["adapter"] = result.get("model_adapter")
    return result


def layer_kinds(config: dict) -> tuple[str, ...]:
    """The per-layer attention kinds the adapter derives.

    The layout comes from the adapter's own normaliser, so this checks
    what the kernel would actually see — not a re-implementation of it.
    """
    normalised = MiniMaxFamilyAdapter().normalize_config(config)
    language = normalised.language
    attn_list = language.get("attn_type_list")
    if attn_list:
        return tuple(
            "full_attention" if int(value) == 1 else "linear_attention"
            for value in attn_list
        )
    return tuple(language.get("layer_types") or ())


def main() -> int:
    families.register_all()

    m2 = fixture("MiniMaxAI-MiniMax-M2")
    m1 = fixture("MiniMaxAI-MiniMax-M1")
    text01 = fixture("MiniMaxAI-MiniMax-Text-01")

    print("=== ① 适配器注册与选择 ===")
    from hcu_train_simulator.models.registry import registered_model_adapters

    names = {a.name for a in registered_model_adapters()}
    check("minimax_family 已注册", "minimax_family" in names)

    for label, config in (("M2", m2), ("M1", m1), ("Text-01", text01)):
        if config is None:
            print(f"SKIP  {label} —— 没有采集到 config")
            continue
        verdict = compatibility.check_compatibility(config)
        check(f"{label} 落到 minimax_family",
              verdict["adapter"] == "minimax_family", str(verdict["adapter"]))
        check(f"{label} 可估算", verdict["supported"] is True,
              str(verdict["reasons"])[:70])

    print("\n=== ② M2 是纯全注意力，不是混合 ===")
    if m2:
        kinds = layer_kinds(m2)
        counts = Counter(kinds)
        check("M2 全部是全注意力层",
              set(counts) == {"full_attention"}, str(dict(counts)))
        check("M2 层数正确", len(kinds) == m2["num_hidden_layers"],
              f"{len(kinds)} vs {m2['num_hidden_layers']}")

        # The released config still carries the field, with its default
        # value. It must not change the estimate.
        without = json.loads(json.dumps(m2))
        without.pop("attn_type_list", None)
        with_list = simulate(m2, "m2-with")
        no_list = simulate(without, "m2-without")
        if with_list and no_list:
            check("带不带 attn_type_list 结果一致",
                  with_list["iteration_ms"] == no_list["iteration_ms"],
                  f"{with_list['iteration_ms']} vs {no_list['iteration_ms']}")
            check("峰值显存一致",
                  with_list["peak_gib"] == no_list["peak_gib"],
                  f"{with_list['peak_gib']} vs {no_list['peak_gib']}")

    print("\n=== ③ M1 / Text-01 是真混合 ===")
    for label, config in (("M1", m1), ("Text-01", text01)):
        if config is None:
            continue
        kinds = layer_kinds(config)
        counts = Counter(kinds)
        check(f"{label} 既有全注意力层也有线性注意力层",
              len(counts) == 2, str(dict(counts)))
        check(f"{label} 线性层占多数",
              counts.get("linear_attention", 0) > counts.get("full_attention", 0),
              str(dict(counts)))
        # Every eighth layer is full attention.
        full_positions = [i for i, k in enumerate(kinds) if k == "full_attention"]
        gaps = {full_positions[i + 1] - full_positions[i]
                for i in range(len(full_positions) - 1)}
        check(f"{label} 全注意力层每隔 8 层", gaps == {8}, str(gaps))

    print("\n=== ④ 近似必须被标注 ===")
    if m2:
        verdict = compatibility.check_compatibility(m2)
        # M2 has no Lightning layer, so the Lightning marker must not fire.
        check("M2 不提 Lightning 近似",
              not any("Lightning" in w for w in verdict["warnings"]),
              "; ".join(verdict["warnings"])[:80])
        # But the genuinely unmodelled parts must be reported.
        check("M2 报告 FP8 量化",
              any("FP8" in w for w in verdict["warnings"]))
        check("M2 报告 sigmoid 路由",
              any("sigmoid" in w for w in verdict["warnings"]))
        check("M2 标记为近似", verdict.get("is_approximate") is True)

    if m1:
        verdict = compatibility.check_compatibility(m1)
        check("M1 报告 Lightning 近似",
              any("Lightning" in w for w in verdict["warnings"]),
              "; ".join(verdict["warnings"])[:80])
        check("M1 标记为近似", verdict.get("is_approximate") is True)

    print("\n=== ⑤ 字段读取正确 ===")
    if m2:
        verdict = compatibility.check_compatibility(m2)
        check("M2 不再因专家数被拒",
              not any("稠密" in r for r in verdict["reasons"]),
              verdict["reasons"][0][:70] if verdict["reasons"] else "")
        check("M2 专家数已被归一化",
              any("归一化" in w for w in verdict["warnings"]))

    print("\n=== ⑥ 估算结果 ===")
    for label, config in (("M2", m2), ("M1", m1)):
        if config is None:
            continue
        result = simulate(config, f"final-{label}")
        if not result:
            check(f"{label} 可仿真", False)
            continue
        check(f"{label} 落到 minimax_family",
              result["adapter"] == "minimax_family", str(result["adapter"]))
        check(f"{label} 迭代时间合理", (result["iteration_ms"] or 0) > 0,
              f"{result['iteration_ms']}ms")
        check(f"{label} 峰值显存合理", (result["peak_gib"] or 0) > 0,
              f"{result['peak_gib']}GiB")

    # The old behaviour treated M2 as a dense decoder: the peak memory
    # came out 132x too small. The adapter has to fix that.
    if m2:
        result = simulate(m2, "final-M2")
        if result:
            check("M2 峰值显存不再按稠密模型算",
                  (result["peak_gib"] or 0) > 1000, f"{result['peak_gib']}GiB")

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL MINIMAX-ADAPTER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
