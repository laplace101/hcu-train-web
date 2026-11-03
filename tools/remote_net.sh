#!/usr/bin/env bash
PY314=/root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14

echo "=== raw connectivity ==="
for url in https://pypi.tuna.tsinghua.edu.cn/simple/ https://mirrors.aliyun.com/pypi/simple/ https://pypi.org/simple/; do
    timeout 10 "$PY314" -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('$url', timeout=7)
    print('  OK      $url', r.status)
except Exception as exc:
    print('  FAIL    $url', type(exc).__name__)
" 2>&1
done

echo
echo "=== DNS ==="
timeout 6 getent hosts pypi.tuna.tsinghua.edu.cn 2>/dev/null || echo "  DNS fail for tsinghua"
timeout 6 getent hosts mirrors.aliyun.com 2>/dev/null || echo "  DNS fail for aliyun"

echo
echo "=== proxy env on this box ==="
env | grep -i proxy || echo "  (no proxy vars)"

echo
echo "=== is there an internal mirror configured anywhere? ==="
grep -rhoE 'index-url[^ ]*' /etc/yum.repos.d /root/.config/pip /root/.pip 2>/dev/null | head -5
grep -rhE 'http://|https://' /etc/yum.repos.d/*.repo 2>/dev/null | grep -oE 'https?://[^ ]+' | head -3