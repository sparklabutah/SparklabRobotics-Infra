#!/usr/bin/env bash
# Start a Jupyter server running Isaac Sim's bundled Python, for VS Code's
# "Existing Jupyter Server..." connection.
#
# Use this when VS Code won't offer the kernelspec directly and asks for a
# server URL instead. The server inherits the correct environment from
# Isaac's own launcher, so its kernels work even where a bare kernelspec
# doesn't.
#
# Prints a http://localhost:<port>/?token=... URL. Paste that into
#   Select Kernel -> Existing Jupyter Server...
# VS Code's Remote-SSH session already tunnels localhost, so no -L forward.
#
#   ./scripts/start_isaac_jupyter.sh [port] [/path/to/isaac-sim-standalone-...]

set -euo pipefail

PORT="${1:-8888}"
ISAAC="${2:-$HOME/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -x "$ISAAC/jupyter_notebook.sh" ]] || {
    echo "ERROR: no jupyter_notebook.sh under $ISAAC" >&2; exit 1; }

# Put the repo on PYTHONPATH before Isaac's setup script runs — it *appends*
# to PYTHONPATH, so anything exported here survives and `import sparklab_sim`
# works in every kernel this server spawns.
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
