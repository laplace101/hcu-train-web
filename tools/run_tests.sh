#!/usr/bin/env bash
# Run the whole verification suite for hcu-train-web.
#
#   tools/run_tests.sh
#
# 1. unit / integration tests for the layered API   (tests/test_api_v1.py)
# 2. image catalogue + launch rendering              (tests/test_launch.py)
# 3. frontend tests against a real jsdom DOM         (tests/frontend_test.js)
# 4. the 35/35 accuracy baseline against the simulator
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-/root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14}"
FAILED=0

run() {
  local label="$1"; shift
  echo ""
  echo "──────────────────────────────────────────────"
  echo "▶ $label"
  echo "──────────────────────────────────────────────"
  if "$@"; then
    echo "✔ $label 通过"
  else
    echo "✘ $label 失败"
    FAILED=1
  fi
}

run "API v1 分层接口"    "$PY" tests/test_api_v1.py
run "镜像与启动命令"      "$PY" tests/test_launch.py

if command -v node >/dev/null 2>&1 && [ -d node_modules/jsdom ]; then
  run "前端 DOM 流程"     node tests/frontend_test.js
else
  echo ""
  echo "⚠ 跳过前端测试：未安装 jsdom（npm install jsdom）"
fi

if [ "${SKIP_BASELINE:-0}" != "1" ]; then
  run "35/35 精度基线"    "$PY" verification/verify_accuracy.py
fi

echo ""
echo "══════════════════════════════════════════════"
if [ "$FAILED" -eq 0 ]; then
  echo "全部测试通过"
else
  echo "存在失败项"
fi
exit "$FAILED"
