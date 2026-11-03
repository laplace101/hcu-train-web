#!/usr/bin/env bash
echo "=== legacy processes (before) ==="
ps -eo pid,ppid,etime,cmd | grep -E 'launch_backend|hcu-train|backend\.py' | grep -v grep

echo
echo "=== stopping them ==="
for pid in $(ps -eo pid,cmd | grep -E 'launch_backend|hcu-train-web/backend\.py' | grep -v grep | awk '{print $1}'); do
    echo "  kill $pid"
    kill "$pid" 2>/dev/null
done
sleep 3
for pid in $(ps -eo pid,cmd | grep -E 'launch_backend|hcu-train-web/backend\.py' | grep -v grep | awk '{print $1}'); do
    echo "  kill -9 $pid (still alive)"
    kill -9 "$pid" 2>/dev/null
done
sleep 1

echo
echo "=== port 7766 now ==="
if (ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null) | grep -q ':7766[[:space:]]'; then
    echo "STILL BUSY:"
    (ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null) | grep ':7766[[:space:]]'
else
    echo "FREE"
fi

echo
echo "=== legacy dir preserved? ==="
ls -d /root/hcu-train-web && echo "yes (untouched, 可随时回退)"