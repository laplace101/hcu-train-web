#!/usr/bin/env bash
cd ${HCU_DEPLOY_DIR:-/root/hcu-train-web-deploy}/hcu-train-web
PY=./.venv/bin/python
export PYTHONIOENCODING=utf-8
export PATH="$PWD/.venv/bin:$PATH"
echo "interpreter: $($PY --version 2>&1)"
echo
$PY -u verification/verify_accuracy.py --refresh 2>&1 | grep -E 'PASS|MISMATCH|ERROR|通过|失败|基线|开始校验'