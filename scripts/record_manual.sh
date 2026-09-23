#!/usr/bin/env bash
# Teleop dataset recording with manual episode boundaries: space starts/ends
# each episode (arms park to zeros on end), r discards it, q quits.
#
#   ./scripts/start_arm_servers.sh          # once, before this
#   ./scripts/record_manual.sh [extra flags for record_manual.py]
#
# Starts a bare relay on :8443 if none is running (torn down on exit); a
# running bare relay is reused. A relay holding the cameras must be stopped
# first — the follower opens the RealSenses itself during recording.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-/opt/miniforge/envs/robot-py312/bin/python}
LOG_DIR=/tmp/yam-teleop-logs
mkdir -p "$LOG_DIR"
cd "$ROOT"

if pgrep -f "relay.server.*--cameras-yaml" > /dev/null; then
    echo "ERROR: a relay with --cameras-yaml is running (teleop_demo.sh?) and" >&2
    echo "holds the RealSense cameras. Stop it, then rerun this script." >&2
    exit 1
fi

PIDS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# ---- relay (bare: no cameras — the follower owns them while recording) ------
if curl -sk --max-time 2 -o /dev/null https://127.0.0.1:8443/; then
    echo "[record] relay already listening on :8443 — reusing it"
else
    echo "[record] starting bare relay (log: $LOG_DIR/relay.log)"
    setsid "$PYTHON_BIN" -m lerobot_robot_sparklab.relay.server --host 0.0.0.0 \
        --ssl-keyfile "$ROOT/key.pem" --ssl-certfile "$ROOT/cert.pem" \
        > "$LOG_DIR/relay.log" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 1 20); do
        curl -sk --max-time 1 -o /dev/null https://127.0.0.1:8443/ && break
        sleep 0.5
    done
    curl -sk --max-time 2 -o /dev/null https://127.0.0.1:8443/ \
        || { echo "[record] relay failed to start — see $LOG_DIR/relay.log"; exit 1; }
fi
LAN_IP=$(ip -4 -o addr show scope global | grep -vE ' (docker|veth|br-|virbr)' \
    | awk '{split($4,a,"/"); print a[1]}' \
    | grep -E '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)' | head -1)
echo "[record] Quest browser → https://${LAN_IP:-<this-host>}:8443/"

# ---- recorder (foreground) --------------------------------------------------
"$PYTHON_BIN" -m lerobot_robot_sparklab.tools.record_manual \
    --robot.type=yam_ultra_bimanual \
    --robot.left_channel=can_left --robot.right_channel=can_right \
    --teleop.type=bi_quest_teleop \
    --teleop.ws_url=wss://127.0.0.1:8443/ws \
    --dataset.repo_id=minhphd/clean \
    --dataset.single_task="clean up the tables" \
    --dataset.num_episodes=100 --dataset.fps=30 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    "$@"
