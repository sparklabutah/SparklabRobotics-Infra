#!/usr/bin/env bash
# Install a self-contained Jupyter kernel for Isaac Sim + sparklab_sim.
#
# Isaac's own kernelspec relies on the environment `jupyter_notebook.sh` sets up
# before launching, so a client that starts kernels itself — VS Code — gets
# "No module named 'isaacsim'". This bakes the resolved environment plus this
# repo's src/ into kernel.json so the kernel stands alone.
#
# Re-run after moving or upgrading Isaac, or moving the repo: every path baked
# in here is absolute.
#
#   ./scripts/install_isaac_kernel.sh [/path/to/isaac-sim-standalone-...]

set -euo pipefail

export ISAAC="${1:-$HOME/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_NAME="sparklab-isaac"

[[ -f "$ISAAC/setup_python_env.sh" ]] || {
    echo "ERROR: no setup_python_env.sh under $ISAAC" >&2
    echo "Pass the Isaac Sim standalone directory as the first argument." >&2
    exit 1
}

export PY="$ISAAC/kit/python/bin/python3"
[[ -x "$PY" ]] || { echo "ERROR: no bundled python at $PY" >&2; exit 1; }

# Resolved in a subshell exactly as Isaac's launcher does, rather than
# reimplementing its long, version-specific path lists.
eval "$(
    # setup_python_env.sh appends to these and assumes an interactive shell
    # where both exist, so `set -u` is dropped in this subshell only.
    set +u
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" PYTHONPATH="${PYTHONPATH:-}"
    export CARB_APP_PATH="$ISAAC/kit" ISAAC_PATH="$ISAAC" EXP_PATH="$ISAAC/apps"
    # shellcheck disable=SC1091
    source "$ISAAC/setup_python_env.sh"
    printf 'RESOLVED_LD=%q\nRESOLVED_PY=%q\n' "$LD_LIBRARY_PATH" "$PYTHONPATH"
)"

# Prepend the repo so `import sparklab_sim` works with no PYTHONPATH gymnastics.
export RESOLVED_LD RESOLVED_PY="$REPO/src:$RESOLVED_PY"

KDIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"
mkdir -p "$KDIR"

"$PY" - "$KDIR/kernel.json" <<'PYEOF'
import json, os, sys

spec = {
    "argv": [os.environ["PY"], "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "SparkLab Isaac Sim (3.12)",
    "language": "python",
    "env": {
        "ISAAC_JUPYTER_KERNEL": "1",
        "CARB_APP_PATH": os.environ["ISAAC"] + "/kit",
        "ISAAC_PATH": os.environ["ISAAC"],
        "EXP_PATH": os.environ["ISAAC"] + "/apps",
        "LD_LIBRARY_PATH": os.environ["RESOLVED_LD"],
        "PYTHONPATH": os.environ["RESOLVED_PY"],
    },
    "metadata": {"debugger": True},
}
with open(sys.argv[1], "w") as f:
    json.dump(spec, f, indent=4)
PYEOF

echo "installed kernel '$KERNEL_NAME' -> $KDIR/kernel.json"
echo "  isaac : $ISAAC"
echo "  repo  : $REPO/src"
echo
echo "Verifying the baked env works with NOTHING inherited from this shell..."
env -i HOME="$HOME" PATH=/usr/bin:/bin \
    CARB_APP_PATH="$ISAAC/kit" ISAAC_PATH="$ISAAC" EXP_PATH="$ISAAC/apps" \
    LD_LIBRARY_PATH="$RESOLVED_LD" PYTHONPATH="$RESOLVED_PY" \
    "$PY" -c "from isaacsim import SimulationApp
import sparklab_sim
print('OK: isaacsim + sparklab_sim import cleanly')"
