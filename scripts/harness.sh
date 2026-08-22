#!/usr/bin/env bash
# Web harness: a high-level agent in front of the low-level VLA.
#
#     ./scripts/harness.sh                      # scripted agent, no API key
#     ./scripts/harness.sh --agent gemini       # needs HARNESS_MODEL + GOOGLE_API_KEY
#     ./scripts/harness.sh --agent none         # UI + REST only, nothing decides
#
# Needs a rollout with a control port already running, because that process owns
# the cameras and the arms and this one deliberately does not:
#
#     ./scripts/start_arm_servers.sh      # terminal 1
#     ./scripts/rollout.sh --mode=live    # terminal 2  (control port 8090)
#     ./scripts/harness.sh                # terminal 3
#
# Override with CONTROL_PORT / PUBLISH_EVERY on the rollout side; PUBLISH_EVERY
# subsamples the JPEG encode off the control loop, since an agent reasoning
# every few seconds does not need a frame per tick.
#
# Then open http://127.0.0.1:8099/ .
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROLLOUT="${ROLLOUT:-http://127.0.0.1:8090}"
PORT="${HARNESS_PORT:-8099}"

python - <<'PY' || { echo "[harness] activate the env holding this package: conda activate robot-py312" >&2; exit 1; }
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("lerobot_robot_sparklab") else 1)
PY

# Warn rather than fail: the UI is useful before the rollout finishes its ~90 s
# model load, and it recovers on its own once the control port appears.
if ! curl -sf --max-time 2 -o /dev/null "$ROLLOUT/status"; then
    echo "[harness] no rollout at $ROLLOUT yet — the UI will show it as down and" >&2
    echo "          reconnect when it comes up (rollout_live --control-port)." >&2
fi

echo "[harness] http://127.0.0.1:$PORT   (rollout $ROLLOUT)"
exec python -m lerobot_robot_sparklab.harness.server \
    --rollout "$ROLLOUT" --port "$PORT" "$@"
