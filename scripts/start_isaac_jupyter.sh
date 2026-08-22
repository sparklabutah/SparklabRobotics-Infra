#!/usr/bin/env bash
# Start a Jupyter server on Isaac Sim's bundled Python, for VS Code's
# "Existing Jupyter Server..." connection.
#
# Use this when VS Code won't offer the kernelspec directly. The server inherits
# its environment from Isaac's own launcher, so its kernels work where a bare
# kernelspec doesn't. Paste the printed URL into
# Select Kernel -> Existing Jupyter Server... — Remote-SSH already tunnels
# localhost, so no -L forward is needed.
#
#   ./scripts/start_isaac_jupyter.sh [port] [/path/to/isaac-sim-standalone-...]

set -euo pipefail

PORT="${1:-8888}"
ISAAC="${2:-$HOME/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -x "$ISAAC/jupyter_notebook.sh" ]] || {
    echo "ERROR: no jupyter_notebook.sh under $ISAAC" >&2; exit 1; }

# Before Isaac's setup script, which appends to PYTHONPATH — so this survives
# into every kernel the server spawns.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

echo "repo on PYTHONPATH : $REPO/src"
echo "isaac              : $ISAAC"
echo "serving notebooks  : $REPO"
echo

cd "$ISAAC"
exec ./jupyter_notebook.sh \
    --no-browser \
    --port "$PORT" \
    --ip 127.0.0.1 \
    --notebook-dir "$REPO"
