#!/usr/bin/env bash
# Full redeploy of hcu-train-web on the workstation, from a fresh bundle.
# Keeps the existing virtualenv (the only expensive-to-rebuild piece) and
# takes a tarball backup of the current app before touching anything.
set -euo pipefail

APP="${HCU_DEPLOY_DIR:-/root/hcu-train-web-deploy}"
ZIP="${HCU_DEPLOY_ZIP:-$APP-new.zip}"
TS=$(date +%Y%m%d-%H%M%S)
BACKUP="${HCU_DEPLOY_BACKUP:-/root/hcu-train-web-backup-$TS.tar.gz}"

echo "=== [1/5] stop service ==="
cd "$APP"
bash deploy.sh stop || true

echo
echo "=== [2/5] backup current app (excluding .venv / runtime) ==="
tar -czf "$BACKUP" --exclude='.venv' --exclude='runtime' \
    -C "$APP" hcu-train-web hcu-train-simulator
ls -la "$BACKUP"

echo
echo "=== [3/5] wipe app files, keep .venv ==="
cd "$APP"
find hcu-train-web -mindepth 1 -maxdepth 1 ! -name '.venv' -exec rm -rf {} +
rm -rf hcu-train-simulator
echo "kept in hcu-train-web/: $(ls -A hcu-train-web | tr '\n' ' ')"

echo
echo "=== [4/5] extract new bundle ==="
unzip -q -o "$ZIP" -d "$APP"
# The zip is produced on Windows, which drops the executable bit.
chmod +x "$APP"/deploy.sh
ls -la "$APP"

echo
echo "=== [5/5] start ==="
cd "$APP"
bash deploy.sh start
sleep 6
bash deploy.sh status
echo "--- health ---"
curl -s -m 20 http://127.0.0.1:7766/api/health
echo