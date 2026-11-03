#!/bin/bash
# Start a Cloudflare quick tunnel exposing local 7766 to a public https URL.
# Writes the URL to /tmp/cf-public-url when ready.

set -e
CF=/opt/cf/cloudflared
LOG=/tmp/cf-tunnel.log
URLFILE=/tmp/cf-public-url

# Cleanup any prior tunnel
for pid in $(pgrep -f "cloudflared tunnel --no-autoupdate"); do
  kill -9 $pid 2>/dev/null || true
done
sleep 1

# Ensure backend is up
/root/hcu-train-web/start.sh start >/dev/null

# Start cloudflared in setsid-detached process
setsid $CF tunnel --no-autoupdate --url http://127.0.0.1:7766 </dev/null > $LOG 2>&1 &
disown

# Wait for URL to appear (up to 15 s)
URL=""
for _ in $(seq 1 30); do
  URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' $LOG 2>/dev/null | head -1 || true)
  [ -n "$URL" ] && break
  sleep 0.5
done

if [ -z "$URL" ]; then
  echo "[cf-tunnel] FAILED to obtain URL — see $LOG"
  exit 1
fi

echo "$URL" > "$URLFILE"
echo "[cf-tunnel] public URL: $URL"
echo "[cf-tunnel] log:        $LOG"
echo "[cf-tunnel] backend:    http://127.0.0.1:7766/"

# Quick sanity probe through the tunnel
sleep 2
echo "[cf-tunnel] /api/health via tunnel:"
curl -s --max-time 10 "$URL/api/health" | head -c 300
echo
