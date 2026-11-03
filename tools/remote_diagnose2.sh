#!/usr/bin/env bash
echo "=== what is the running process (7766) ==="
for pid in 1484652; do
    if [ -d "/proc/$pid" ]; then
        echo "cmdline: $(tr '\0' ' ' < /proc/$pid/cmdline)"
        echo "exe    : $(readlink -f /proc/$pid/exe)"
    else
        echo "pid $pid is gone"
    fi
done

echo
echo "=== uv / local pythons under /root/.local ==="
ls -d /root/.local/share/uv/python/*/bin/python3* 2>/dev/null || echo "(no uv pythons)"
ls -d /root/.local/bin/* 2>/dev/null | head
ls -d /root/.venv /root/venv /opt/*venv* 2>/dev/null || echo "(no venvs in usual spots)"

echo
echo "=== find any python >= 3.8 on the box ==="
found=0
for c in /usr/bin/python3.* /usr/local/bin/python3.* /opt/*/bin/python3* /root/*/bin/python3* /usr/lib/*/bin/python3*; do
    [ -x "$c" ] || continue
    case "$c" in
        *python3.7|*.so|*-config|*.a) continue ;;
    esac
    ver=$("$c" --version 2>&1)
    echo "  $c -> $ver"
    found=1
done
[ "$found" = "0" ] && echo "  (only 3.7 found)"

echo
echo "=== is the simulator source 3.7-compatible at all? (scan for PEP585/604) ==="
if [ -d /root/hcu-train-simulator/src ]; then
    echo "files using builtin generics in runtime positions:"
    grep -rEn '(^|[^a-zA-Z_])(tuple|list|dict|set)\[' /root/hcu-train-simulator/src/hcu_train_simulator --include=*.py \
        | grep -vE '#|"""|typing' | head -8
    echo "..."
    echo "total matches: $(grep -rEc '(^|[^a-zA-Z_])(tuple|list|dict|set)\[' /root/hcu-train-simulator/src/hcu_train_simulator --include=*.py 2>/dev/null | awk -F: '{s+=$2} END {print s}')"
fi

echo
echo "=== yum: can we install a newer python? ==="
yum repolist 2>/dev/null | tail -5
echo "--- available python39/311 packages ---"
yum list available 2>/dev/null | grep -Ei '^(python3(9|10|11|12)|python39)\b' | head -10 || echo "(none listed)"
echo "--- python3-devel / platform-python ---"
yum list available 2>/dev/null | grep -Ei 'platform-python|python3-devel' | head -5 || true