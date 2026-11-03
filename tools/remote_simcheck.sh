#!/usr/bin/env bash
APP=${HCU_DEPLOY_DIR:-/root/hcu-train-web-deploy}/hcu-train-web
echo "=== process alive? ==="
if [ -f "$APP/runtime/server.pid" ]; then
    pid=$(cat "$APP/runtime/server.pid")
    kill -0 "$pid" 2>/dev/null && echo "alive pid=$pid" || echo "DEAD pid=$pid"
else
    echo "no pidfile"
fi
echo
echo "=== listeners ==="
ss -lntp 2>/dev/null | grep 7766 || echo "(port not listening)"
echo
echo "=== server.log tail ==="
tail -n 40 "$APP/runtime/server.log" 2>/dev/null || echo "(no log)"
echo
echo "=== local curl of /api/simulate (direct on workstation) ==="
time curl -sS -m 180 -o /tmp/sim.json -w "http=%{http_code} bytes=%{size_download}\n" \
  -X POST http://127.0.0.1:7766/api/simulate \
  -H 'Content-Type: application/json' \
  -d '{"task":"full","hardware":"bw1000","operator_profile":"bw1000","model":"qwen3-8b","parallel":{"tp_size":8,"pp_size":1,"cp_size":1,"ep_size":1,"num_gpus":8,"seq_length":4096,"micro_batch_size":1,"global_batch_size":256,"use_distributed_optimizer":true,"sequence_parallel":true,"full_recompute":false,"use_fp8_training":false}}' 2>&1
echo "--- response head ---"
head -c 400 /tmp/sim.json 2>/dev/null; echo