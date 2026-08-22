#!/usr/bin/env bash
# Run something under Isaac Sim's bundled Python, with this repo importable.
#
# There is NO conda environment for sim work. Isaac ships its own Python 3.12
# and cannot be installed into a conda env; its own python.sh warns if you run
# it from inside one. This wrapper makes that a non-issue: it clears any active
# conda env, puts <repo>/src on PYTHONPATH, and hands off.
#
#   ./scripts/isaac_python.sh -m sparklab_sim.convert    # module
#   ./scripts/isaac_python.sh my_script.py               # script
#   ./scripts/isaac_python.sh                            # REPL
#
# Override the install with ISAAC=/path/to/isaac-sim-standalone-...

set -euo pipefail

ISAAC="${ISAAC:-$HOME/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -x "$ISAAC/python.sh" ]] || {
    echo "ERROR: no python.sh under $ISAAC" >&2
    echo "Set ISAAC=/path/to/isaac-sim-standalone-... and retry." >&2
    exit 1
}

# Leave any conda env. python.sh only warns, but a conda env's LD_LIBRARY_PATH
# and site-packages can shadow Isaac's and produce failures far from the cause
# (e.g. "SRE module mismatch" from a mismatched stdlib).
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    while [[ -n "${CONDA_PREFIX:-}" ]]; do conda deactivate || break; done
fi

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ISAAC/python.sh" "$@"
