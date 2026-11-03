#!/usr/bin/env bash
# Deploy hcu-train-web on a Linux workstation / server.
#
#   ./deploy.sh                 # default 0.0.0.0:7766
#   PORT=8080 ./deploy.sh       # custom port
#   VENV=0 ./deploy.sh          # install into the system Python instead of ./.venv
#
# The bundle is self-contained: hcu-train-simulator sits next to hcu-train-web,
# so nothing has to be cloned or installed from the network except pip deps.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The bundle ships the app in hcu-train-web/ with hcu-train-simulator/ as its
# sibling. Allow the script to also be dropped inside the app directory itself.
if [ -f "$HERE/hcu-train-web/app.py" ]; then
    APP_DIR="$HERE/hcu-train-web"
    SIM_DIR="$HERE/hcu-train-simulator"
elif [ -f "$HERE/app.py" ]; then
    APP_DIR="$HERE"
    SIM_DIR="$HERE/../hcu-train-simulator"
else
    APP_DIR="$HERE/hcu-train-web"
    SIM_DIR="$HERE/hcu-train-simulator"
fi

cd "$APP_DIR"

ACTION="${1:-start}"
PORT="${PORT:-7766}"
HOST="${HOST:-0.0.0.0}"
VENV="${VENV:-1}"
PY="${PYTHON:-python3}"
PIDFILE="$APP_DIR/runtime/server.pid"
LOGFILE="$APP_DIR/runtime/server.log"

log()  { printf '\033[36m[hcu-train-web]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[hcu-train-web]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[hcu-train-web]\033[0m %s\n' "$*" >&2; exit 1; }

running_pid() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && cat "$PIDFILE"; }

stop_server() {
    local pid
    if pid=$(running_pid); then
        log "停止服务 (pid $pid) ..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$PIDFILE"
        log "已停止"
    else
        log "服务未在运行"
        rm -f "$PIDFILE"
    fi
}

case "$ACTION" in
    stop)
        mkdir -p runtime
        stop_server
        exit 0
        ;;
    restart)
        mkdir -p runtime
        stop_server
        ACTION=start
        ;;
    status)
        if pid=$(running_pid); then
            echo "运行中 (pid $pid)  http://${HOST}:${PORT}/"
            exit 0
        fi
        echo "未运行"
        exit 1
        ;;
    start) ;;
    *) die "未知参数: $ACTION（可用: start | stop | restart | status）" ;;
esac

# ---------------------------------------------------------------- preflight
# The reference simulator needs Python >= 3.9 (it evaluates PEP 585/604
# annotations such as `tuple[str, str]` at import time). Many site images ship
# only 3.7, so probe for a usable interpreter rather than trusting `python3`.
find_python() {
    if [ -n "$PY" ] && [ "$PY" != "python3" ] && command -v "$PY" >/dev/null 2>&1; then
        echo "$PY"; return 0
    fi
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
            echo "$candidate"; return 0
        fi
    done
    # Fall back to self-installed interpreters (uv / pyenv / conda layouts).
    for candidate in \
        "$HOME"/.local/share/uv/python/*/bin/python3 \
        "$HOME"/.local/share/uv/python/*/bin/python3.1* \
        /opt/conda/bin/python3 /root/miniconda3/bin/python3 /root/anaconda3/bin/python3 \
        /usr/local/bin/python3.1* ; do
        [ -x "$candidate" ] || continue
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
            echo "$candidate"; return 0
        fi
    done
    return 1
}

RESOLVED_PY=$(find_python) || die \
"未找到 Python >= 3.9。参考仓库使用了 PEP 585/604 注解，Python 3.7/3.8 无法导入。
 可用方式：安装新版本，或用 uv 装一个：  curl -LsSf https://astral.sh/uv/install.sh | sh
 然后重新执行本脚本（或用 PY=/path/to/python3.12 ./deploy.sh 指定解释器）"

[ -f "$APP_DIR/app.py" ] || die "app.py 不存在（期望路径：$APP_DIR/app.py）"
[ -d "$SIM_DIR/src/hcu_train_simulator" ] || die "缺少 $SIM_DIR/src/hcu_train_simulator"

log "Python: $RESOLVED_PY ($("$RESOLVED_PY" --version 2>&1))"

# -------------------------------------------------------- pip index (CN-safe)
# Many intranet/greatfirewall machines cannot reach pypi.org directly but can
# reach domestic mirrors, so default to one and allow an explicit override.
INDEX_ARGS=""
case "${PIP_INDEX_URL:-}" in
    "") INDEX_ARGS="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn" ;;
    default|"") ;;
    *) INDEX_ARGS="-i $PIP_INDEX_URL" ;;
esac

# ------------------------------------------------------------ python env
if [ "$VENV" = "1" ]; then
    if [ ! -d "$APP_DIR/.venv" ]; then
        log "创建虚拟环境 .venv (基于 $RESOLVED_PY) ..."
        "$RESOLVED_PY" -m venv "$APP_DIR/.venv" 2>/dev/null || die \
"创建 venv 失败（缺少 ensurepip / python3-venv）。可以改用系统解释器直接安装： VENV=0 ./deploy.sh"
    fi
    PY="$APP_DIR/.venv/bin/python"
else
    PY="$RESOLVED_PY"
fi

log "安装依赖 ${INDEX_ARGS:-(使用 pip 默认源)} ..."
# shellcheck disable=SC2086
"$PY" -m pip install --quiet --upgrade pip $INDEX_ARGS >/dev/null 2>&1 || true
# shellcheck disable=SC2086
"$PY" -m pip install --quiet -r requirements.txt $INDEX_ARGS || die "依赖安装失败（可换源重试：PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple ./deploy.sh）"

# --------------------------------------------------------------- start
mkdir -p runtime
stop_server

# Refuse to fight an unknown process for the port: report who holds it instead.
port_busy() {
    (ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null) | grep -q ":${PORT}[[:space:]]"
}
if port_busy; then
    warn "端口 ${PORT} 已被占用，当前占用进程："
    (ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null) | grep ":${PORT}[[:space:]]" | sed 's/^/    /'
    die "请先停掉该进程，或换端口：  PORT=8080 ./deploy.sh"
fi

log "启动服务 ${HOST}:${PORT} ..."
nohup "$PY" -u app.py --host "$HOST" --port "$PORT" >"$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

# -------------------------------------------------------------- health
OK=0
for _ in $(seq 1 30); do
    sleep 1
    if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
        OK=1
        break
    fi
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
done

if [ "$OK" != "1" ]; then
    warn "服务未能在 30 秒内就绪，最后 30 行日志："
    tail -n 30 "$LOGFILE" || true
    die "部署失败"
fi

MODEL_COUNT=$("$PY" -c "
import json, pathlib
d = json.loads(pathlib.Path('catalog/catalog.json').read_text(encoding='utf-8'))
m = d['models']
print(f\"{sum(1 for x in m if x.get('verified'))}/{len(m)}\")
" 2>/dev/null || echo "?")

IPS=$(hostname -I 2>/dev/null || ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)
SHORT=$(hostname -s 2>/dev/null || echo "工作站")

echo
log "部署成功"
echo "  本机自检   : http://127.0.0.1:${PORT}/"
echo "  已校验模型 : ${MODEL_COUNT}"
echo "  PID        : $(cat "$PIDFILE")   日志: $LOGFILE"
for ip in $IPS; do
    echo "  局域网访问 : http://${ip}:${PORT}/"
done
echo
echo "  在你自己的电脑上打开上面任意一个「局域网访问」地址即可。"
echo "  如果打不开，按顺序排查："
echo "    1) 确认两台机器在同一内网： ping ${SHORT}"
echo "    2) 放行端口： firewall-cmd --add-port=${PORT}/tcp --permanent && firewall-cmd --reload"
echo "    3) 或改用 SSH 隧道： ssh -L ${PORT}:127.0.0.1:${PORT} <用户>@<工作站IP>"
echo "       然后在本地浏览器打开 http://127.0.0.1:${PORT}/"
echo
echo "  常用命令:  ./deploy.sh stop | ./deploy.sh status | tail -f $LOGFILE"