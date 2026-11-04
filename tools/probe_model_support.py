"""Batch-test model support across families, using real ModelScope configs.

For each model the script:
  1. fetches the real config.json from ModelScope (read-only),
  2. runs the compatibility check,
  3. actually runs the simulator, and records what happened.

The simulator's result is the ground truth: the compatibility check is a
prediction of it, and any disagreement is worth knowing about.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tools/probe_model_support.py [family ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine import config_builder  # noqa: E402
from estimators import adapter  # noqa: E402
from services import compatibility  # noqa: E402
from services import modelscope as ms  # noqa: E402

#: The models to probe, grouped by family. The repo ids are the ones on
#: ModelScope; a repo that has moved is reported as unreachable rather than
#: silently skipped.
#: Models are fetched from ModelScope where it hosts them, and from the
#: Hugging Face mirror otherwise — the config.json is the same file either
#: way. The source is recorded per model so the report is unambiguous.
MODELS: dict[str, list[tuple[str, str]]] = {
    "qwen": [
        ("Qwen/Qwen3-8B", "ms"),
        ("Qwen/Qwen3-30B-A3B", "ms"),
        ("Qwen/Qwen3-235B-A22B", "ms"),
        ("Qwen/Qwen3-VL-8B-Instruct", "ms"),
        ("Qwen/Qwen3-Next-80B-A3B-Instruct", "ms"),
        ("Qwen/Qwen3.5-35B-A3B", "ms"),
        ("Qwen/Qwen2.5-72B-Instruct", "ms"),
        ("Qwen/Qwen3-14B", "ms"),
    ],
    "deepseek": [
        ("deepseek-ai/DeepSeek-V3", "ms"),
        ("deepseek-ai/DeepSeek-V2", "ms"),
        ("deepseek-ai/DeepSeek-V3.1", "ms"),
        ("deepseek-ai/DeepSeek-V3.2-Exp", "ms"),
        ("deepseek-ai/DeepSeek-R1", "ms"),
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "ms"),
    ],
    "glm": [
        ("zai-org/GLM-4.5", "hf"),
        ("zai-org/GLM-4.6V-Flash", "hf"),
        ("THUDM/glm-4-9b-chat-hf", "hf"),
        ("glm-5", "catalog"),
    ],
    "minimax": [
        ("MiniMaxAI/MiniMax-M2", "hf"),
        ("MiniMaxAI/MiniMax-M1", "hf"),
        ("MiniMaxAI/MiniMax-Text-01", "hf"),
    ],
    "other": [
        ("mistralai/Mistral-7B-Instruct-v0.3", "ms"),
        ("microsoft/phi-3-mini-4k-instruct", "ms"),
        ("moonshotai/Kimi-K2-Instruct", "ms"),
        ("google/gemma-3-12b-it", "ms"),
        ("meta-llama/Llama-3.1-8B-Instruct", "hf"),
    ],
}

#: The parallel layout used for the probe: 8 cards, a modest batch, a
#: sequence length every model can hold.
PROBE_PARALLEL = {
    "tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
    "num_gpus": 8, "micro_batch_size": 1, "global_batch_size": 256,
    "seq_length": 4096,
}


#: The proxy the local environment reaches the Hugging Face mirror through.
#: A background process does not necessarily inherit it, and the mirror is
#: unreachable without it — so the address is resolved here rather than
#: being left to the environment.
HF_PROXY_CANDIDATES = (
    "http://127.0.0.1:10808",
    "http://localhost:10808",
)


def _proxy_for_hf() -> str | None:
    """The proxy to use for the Hugging Face mirror, if one is reachable."""
    import os
    import urllib.request

    candidates = [os.environ.get("HTTPS_PROXY"), os.environ.get("HTTP_PROXY")]
    candidates.extend(HF_PROXY_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": candidate, "https": candidate}))
            with opener.open("https://huggingface.co/api/models/gpt2", timeout=8):
                return candidate
        except Exception:  # noqa: BLE001
            continue
    return None


def _fetch_hf(repo_id: str) -> tuple[dict, str | None]:
    """Fetch a config from the Hugging Face mirror."""
    import urllib.request

    proxy = _proxy_for_hf()
    handlers = {"http": proxy, "https": proxy} if proxy else {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(handlers))
    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "dcu-train-web/2.0"})
    with opener.open(req, timeout=60) as response:
        return json.loads(response.read().decode()), None


def probe(repo_id: str, source: str = "ms") -> dict:
    """Fetch, check and simulate one model."""
    record: dict = {"repo": repo_id, "source": source}

    if source == "catalog":
        path = PROJECT_ROOT / "catalog" / "models" / repo_id / "config.json"
        if not path.exists():
            record["status"] = "unreachable"
            record["detail"] = "目录中没有该模型"
            return record
        config = json.loads(path.read_text(encoding="utf-8"))
        record["revision"] = "catalog"
    elif source == "hf":
        try:
            config, _ = _fetch_hf(repo_id)
        except Exception as exc:  # noqa: BLE001
            record["status"] = "unreachable"
            record["detail"] = f"{type(exc).__name__}: {exc}"[:120]
            return record
        record["revision"] = "hf-main"
    else:
        try:
            resolved = ms.fetch_modelscope_config(repo_id)
        except ms.ModelSourceError as exc:
            record["status"] = "unreachable"
            record["detail"] = str(exc)[:120]
            return record
        config = resolved.config
        record["revision"] = resolved.revision
    record["model_type"] = config.get("model_type")
    record["architectures"] = config.get("architectures")

    verdict = compatibility.check_compatibility(config)
    record["adapter"] = verdict["adapter"]
    record["adapter_kind"] = ("specialised" if verdict.get("adapter_is_specialised")
                              else "fallback" if verdict.get("adapter_is_fallback")
                              else "unknown")
    record["check_supported"] = verdict["supported"]
    record["check_reasons"] = verdict["reasons"]

    # Write the config out so the simulator can read it.
    target_dir = PROJECT_ROOT / "runtime" / "probe"
    target_dir.mkdir(parents=True, exist_ok=True)
    safe = repo_id.replace("/", "-")
    path = target_dir / f"{safe}" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    started = time.time()
    try:
        sim_config = config_builder.build_config(
            str(path),
            task="full", hardware="bw1000", operator_profile="bw1000",
            parallel=dict(PROBE_PARALLEL),
        )
        result = adapter.summarise_simulation(adapter.simulate(sim_config))
        record["status"] = "ok"
        record["iteration_ms"] = result.get("iteration_ms")
        record["tgs"] = result.get("tgs")
        record["peak_gib"] = result.get("peak_gib")
        record["adapter_used"] = result.get("model_adapter")
    except Exception as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["detail"] = f"{type(exc).__name__}: {exc}"[:300]
    record["elapsed_s"] = round(time.time() - started, 1)
    return record


def main() -> int:
    families = sys.argv[1:] or list(MODELS)
    results: list[dict] = []

    for family in families:
        repos = MODELS.get(family)
        if not repos:
            print(f"未知系列：{family}", file=sys.stderr)
            continue
        print(f"\n{'=' * 70}\n{family}\n{'=' * 70}")
        for repo, source in repos:
            record = probe(repo, source)
            results.append(record)
            status = record["status"]
            mark = {"ok": "✓", "failed": "✗", "unreachable": "?"}.get(status, "?")
            print(f"  {mark} {repo}")
            print(f"      model_type={record.get('model_type')} "
                  f"arch={record.get('architectures')}")
            print(f"      适配器={record.get('adapter')} "
                  f"({record.get('adapter_kind')}) "
                  f"兼容性检查={'通过' if record.get('check_supported') else '拒绝'}")
            if status == "ok":
                print(f"      仿真成功: iteration={record['iteration_ms']}ms "
                      f"tgs={record['tgs']} peak={record['peak_gib']}GiB "
                      f"用时={record['elapsed_s']}s")
                print(f"      内核实际使用适配器={record.get('adapter_used')}")
            elif status == "failed":
                print(f"      仿真失败: {record['detail']}")
            else:
                print(f"      不可达: {record['detail']}")
            for reason in record.get("check_reasons") or []:
                print(f"        · {reason}")

    out = PROJECT_ROOT / "runtime" / "probe" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n\n{'=' * 70}")
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]
    unreachable = [r for r in results if r["status"] == "unreachable"]
    print(f"可计算 {len(ok)} / 不可计算 {len(failed)} / 不可达 {len(unreachable)}")
    print(f"结果写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
