"""Verify the deployed workstation instance from the local PC."""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://192.168.7.62:7766"

# The Windows console defaults to GBK; never let a stray symbol abort the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - very old interpreters
    pass


def call(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read())
    return body, round((time.time() - started) * 1000)


print("=== health ===")
health, ms = call("/api/health")
print(f"  {health}  ({ms} ms)")

print("\n=== bootstrap ===")
boot, ms = call("/api/bootstrap")
verified = sum(1 for m in boot["models"] if m.get("verified"))
print(f"  models={len(boot['models'])} verified={verified} ({ms} ms)")
print(f"  simulator={boot['simulator']['repo']}")

print("\n=== simulate Qwen3-8B (tp=8, 8 GPUs) ===")
payload = {
    "task": "full", "hardware": "bw1000", "operator_profile": "bw1000", "model": "qwen3-8b",
    "parallel": {"tp_size": 8, "pp_size": 1, "cp_size": 1, "ep_size": 1, "num_gpus": 8,
                 "seq_length": 4096, "micro_batch_size": 1, "global_batch_size": 256,
                 "use_distributed_optimizer": True, "sequence_parallel": True,
                 "full_recompute": False, "use_fp8_training": False},
}
result, ms = call("/api/simulate", payload)
s = result["result"]["summary"]
print(f"  ok={result['ok']} ({ms} ms)")
print(f"  参数量     {s['model_size_b']} B")
print(f"  迭代时间   {s['iteration_ms']} ms")
print(f"  TGS        {s['tgs']}")
print(f"  MFU        {s['mfu_percent']}%")
print(f"  峰值显存   {result['result']['peak_memory_gib']} GiB / {result['result']['hbm_gib']} GiB")
print(f"  适配器     {result['result']['model_adapter']}")

print("\n  与本地 Windows 部署对比（同一配置）:")
print("    Windows : iteration=68309.7 ms  tgs=1918.79  mfu=21.05%  mem=21.88 GiB")
same = (abs(s["iteration_ms"] - 68309.7) < 0.5 and abs(s["tgs"] - 1918.79) < 0.5
        and abs(s["mfu_percent"] - 21.05) < 0.01)
print(f"    Linux   : iteration={s['iteration_ms']} ms  tgs={s['tgs']}  mfu={s['mfu_percent']}%  "
      f"mem={result['result']['peak_memory_gib']} GiB")
print(f"    一致性  : {'完全一致 PASS' if same else '存在差异 FAIL'}")

print("\n=== solve Qwen3-8B ===")
sol, ms = call("/api/solve", payload)
if sol["ok"]:
    for label in ("minimum", "performance"):
        row = sol["result"].get(label)
        if row:
            p, perf = row["parallel"], row["performance"]
            print(f"  {label:12s} {p['num_gpus']:>3}卡  TP{p['tp_size']} PP{p['pp_size']} DP{p['dp_size']}"
                  f"  峰值 {row['peak_memory_gib']} GiB  iter {perf['iteration_ms']} ms"
                  f"  TGS {perf['tgs']}  MFU {round(perf['mfu'] * 100, 2)}%")
else:
    print(f"  solve failed: {sol.get('error')}")