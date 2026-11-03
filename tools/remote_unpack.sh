#!/usr/bin/env bash
set -e
cd /root
rm -rf ${HCU_DEPLOY_DIR:-/root/hcu-train-web-deploy}
unzip -q -o /root/hcu-train-web-deploy.zip -d ${HCU_DEPLOY_DIR:-/root/hcu-train-web-deploy}
echo "=== unpacked ==="
ls ${HCU_DEPLOY_DIR:-/root/hcu-train-web-deploy}
echo
echo "=== app tree (top 2 levels) ==="
find /root/hcu-train-web-deploy -maxdepth 2 | sort | head -30
echo
echo "=== critical files present? ==="
for f in deploy.sh deploy.cmd DEPLOY.md \
         hcu-train-web/app.py hcu-train-web/requirements.txt \
         hcu-train-web/catalog/catalog.json \
         hcu-train-web/engine/engine.py \
         hcu-train-simulator/src/hcu_train_simulator/simulator.py \
         hcu-train-simulator/src/hcu_train_simulator/profiles/builtin/manifest.yaml \
         hcu-train-simulator/src/hcu_train_simulator/communication/bandwidth/bw1000_1.txt \
         hcu-train-simulator/src/hcu_train_simulator/templates/config.yaml; do
    p="${HCU_DEPLOY_DIR:-/root/hcu-train-web-deploy}/$f"
    if [ -f "$p" ]; then echo "  OK   $f"; else echo "  MISS $f"; fi
done
echo
echo "vendored model configs: $(find ${HCU_DEPLOY_DIR:-/root/hcu-train-web-deploy}/hcu-train-web/catalog/models -name config.json 2>/dev/null | wc -l)"
chmod +x ${HCU_DEPLOY_DIR:-/root/hcu-train-web-deploy}/deploy.sh
echo "deploy.sh executable: $([ -x ${HCU_DEPLOY_DIR:-/root/hcu-train-web-deploy}/deploy.sh ] && echo yes || echo no)"