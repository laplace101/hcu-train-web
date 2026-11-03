#!/usr/bin/env bash
PY314=/root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
[ -x "$PY314" ] || PY314=/root/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/bin/python3.14

echo "=== interpreter ==="
echo "PY314=$PY314"
"$PY314" --version

echo
echo "=== can it import the simulator? ==="
PYTHONPATH=/root/hcu-train-simulator/src "$PY314" -c "
import hcu_train_simulator.simulator as s
print('simulator import: OK')
from hcu_train_simulator.commands.solve import solve_training_config
from hcu_train_simulator.simulator import run_simulation
print('entrypoints import: OK')
" 2>&1 | tail -6

echo
echo "=== web deps present in this interpreter? ==="
"$PY314" - <<'EOF'
import importlib.util as u
for mod in ("flask", "yaml", "tabulate", "requests", "werkzeug", "jinja2"):
    print(f"  {mod:10s} {'YES' if u.find_spec(mod) else 'no'}")
EOF

echo
echo "=== site-packages location ==="
"$PY314" -c "import site,sys; print('\n'.join(site.getsitepackages())); print('prefix:', sys.prefix)"

echo
echo "=== is pip usable offline for this interpreter? ==="
"$PY314" -m pip --version 2>&1 | head -1

echo
echo "=== wheels already on disk we could reuse? ==="
find /root /opt /tmp -maxdepth 4 -name "*.whl" 2>/dev/null | head -10 || echo "(none)"
find /root -maxdepth 4 -type d -name "flask*" 2>/dev/null | head -5 || true