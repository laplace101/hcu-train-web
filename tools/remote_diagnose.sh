#!/usr/bin/env bash
echo "=== what is listening on 7766 ==="
if command -v ss >/dev/null 2>&1; then
    ss -lntp 2>/dev/null | grep ':7766' || echo "(gone)"
fi
for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    if [ -r "/proc/$pid/cmdline" ]; then
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
        case "$cmd" in
            *app.py*|*hcu*) echo "pid=$pid  $cmd" ;;
        esac
    fi
done | head -10

echo
echo "=== existing /root/hcu-train-web ==="
ls -la /root/hcu-train-web 2>/dev/null | head -20
echo "--- is there a new-style app.py? ---"
[ -f /root/hcu-train-web/app.py ] && echo "YES (new layout)" || echo "no app.py (legacy layout)"
[ -f /root/hcu-train-web/backend.py ] && echo "has legacy backend.py" || true
[ -d /root/hcu-train-web/catalog ] && echo "has catalog/" || echo "no catalog/ (legacy)"

echo
echo "=== existing /root/hcu-train-simulator ==="
ls /root/hcu-train-simulator 2>/dev/null | head -12
if [ -d /root/hcu-train-simulator/src/hcu_train_simulator ]; then
    echo "src present"
    [ -f /root/hcu-train-simulator/src/hcu_train_simulator/commands/solve.py ] && echo "has solve.py (recent version)"
else
    echo "NO src/ layout"
fi

echo
echo "=== other pythons available ==="
for p in python3.6 python3.8 python3.9 python3.10 python3.11 python3.12 python3.13; do
    command -v "$p" >/dev/null 2>&1 && echo "  $p -> $($p --version 2>&1)"
done
ls /usr/bin/python3.* /usr/local/bin/python3.* 2>/dev/null | sort -u
command -v conda >/dev/null 2>&1 && echo "conda present"
ls -d /opt/conda /root/miniconda3 /root/anaconda3 2>/dev/null

echo
echo "=== can the simulator import on python3.7? ==="
if [ -d /root/hcu-train-simulator/src ]; then
    PYTHONPATH=/root/hcu-train-simulator/src python3 -c "
import hcu_train_simulator.simulator as s
print('import simulator: OK')
" 2>&1 | tail -5
fi

echo
echo "=== pip / internet ==="
timeout 15 python3 -m pip download --no-deps --dest /tmp/piptest tabulate==0.10.0 >/dev/null 2>&1 \
    && echo "pip download: OK" || echo "pip download: FAILED (offline or blocked)"
rm -rf /tmp/piptest 2>/dev/null

echo
echo "=== pip index config ==="
cat /etc/pip.conf 2>/dev/null || echo "(no /etc/pip.conf)"
cat ~/.pip/pip.conf 2>/dev/null || echo "(no ~/.pip/pip.conf)"
cat ~/.config/pip/pip.conf 2>/dev/null || echo "(no ~/.config/pip/pip.conf)"