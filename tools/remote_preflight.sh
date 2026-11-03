#!/usr/bin/env bash
echo "=== identity ==="
echo "user=$(whoami)  home=$HOME"
echo "host=$(hostname)"

echo "=== os ==="
cat /etc/os-release 2>/dev/null | head -3
echo "arch=$(uname -m)"

echo "=== python ==="
for p in python3 python; do
    if command -v "$p" >/dev/null 2>&1; then
        echo "$p -> $($p --version 2>&1)"
    fi
done
if command -v python3 >/dev/null 2>&1; then
    echo "pip: $(python3 -m pip --version 2>&1 | head -1)"
    python3 - <<'EOF'
try:
    import venv
    print("venv module: OK")
except Exception as exc:
    print(f"venv module: MISSING ({exc})")
import sys
print("python >= 3.9:", sys.version_info >= (3, 9), sys.version.split()[0])
EOF
fi

echo "=== disk ==="
df -h "$HOME" 2>/dev/null | tail -2

echo "=== package manager ==="
for m in yum dnf apt-get; do
    command -v "$m" >/dev/null 2>&1 && echo "have: $m"
done

echo "=== firewall ==="
command -v firewall-cmd >/dev/null 2>&1 && echo "firewalld: yes" || echo "firewalld: no"
command -v iptables >/dev/null 2>&1 && echo "iptables: yes" || echo "iptables: no"

echo "=== network ==="
echo "IPs: $(hostname -I 2>/dev/null)"
echo "existing listeners on 7766:"
(ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E ':7766|:7767' || echo "  (port 7766 free)"

echo "=== existing installs ==="
ls -d "$HOME"/hcu-train* 2>/dev/null || echo "  (no prior hcu-train-web in \$HOME)"
ls -d /opt/hcu-train* 2>/dev/null || true

echo "=== outbound net (pip reachability) ==="
timeout 12 python3 -c "
import urllib.request
for url in ('https://pypi.org/simple/', 'https://pypi.tuna.tsinghua.edu.cn/simple/'):
    try:
        urllib.request.urlopen(url, timeout=6)
        print('reachable:', url)
    except Exception as exc:
        print('unreachable:', url, type(exc).__name__)
" 2>&1 | head -4