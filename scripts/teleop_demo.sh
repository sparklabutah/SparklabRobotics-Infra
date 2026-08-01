#!/usr/bin/env bash
# Launch the full YAM-Ultra teleop stack in one terminal:
#   1. relay (TLS, reused if one is already listening on :8443)
#   2. MuJoCo viewers for both arms (skipped when there is no DISPLAY)
#   3. the bimanual hardware bridge (foreground; its log is your console)
#
# Usage:
#   ./scripts/teleop_demo.sh [args forwarded to teleop_bimanual.py]
#   ./scripts/teleop_demo.sh --sim              # no hardware dry-run
#   ./scripts/teleop_demo.sh --left-channel can_right --right-channel can_left
#
# Ctrl-C stops the bridge (it ramps the arms home first), then tears down
# the viewers/relay this script started. A second Ctrl-C skips the ramp.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PKG="$ROOT/src/lerobot_robot_sparklab"
ENV_BIN=/opt/miniforge/envs/teleop/bin
WS_URL=wss://127.0.0.1:8443/ws
LOG_DIR=/tmp/yam-teleop-logs
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# ---- relay -----------------------------------------------------------------
if curl -sk --max-time 2 -o /dev/null https://127.0.0.1:8443/; then
    echo "[run] relay already listening on :8443 — reusing it"
else
    echo "[run] starting relay (log: $LOG_DIR/relay.log)"
    setsid env VR_TELEOP_CAMERAS_YAML="$PKG/robots/yam_ultra/config/cameras.yaml" \
        "$ENV_BIN/vr-teleop-relay" --host 0.0.0.0 \
        --ssl-keyfile "$ROOT/certs/key.pem" --ssl-certfile "$ROOT/certs/cert.pem" \
        > "$LOG_DIR/relay.log" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 1 20); do
        curl -sk --max-time 1 -o /dev/null https://127.0.0.1:8443/ && break
        sleep 0.5
    done
    curl -sk --max-time 2 -o /dev/null https://127.0.0.1:8443/ \
        || { echo "[run] relay failed to start — see $LOG_DIR/relay.log"; exit 1; }
fi
LAN_IP=$(hostname -I | awk '{print $1}')
echo "[run] Quest browser → https://$LAN_IP:8443/"

# ---- viewers ---------------------------------------------------------------
if [ -n "${DISPLAY:-}" ]; then
    for arm in right left; do
        echo "[run] starting $arm-arm viewer (log: $LOG_DIR/viewer_$arm.log)"
        setsid "$ENV_BIN/python" "$PKG/tools/viewer_client.py" \
            --url "$WS_URL" --arm "$arm" > "$LOG_DIR/viewer_$arm.log" 2>&1 &
        PIDS+=($!)
    done
else
    echo "[run] no DISPLAY — skipping MuJoCo viewers (run at the workstation,"
    echo "      or DISPLAY=:0 ./scripts/teleop_demo.sh to use its monitor)"
fi

# ---- bridge (foreground — Ctrl-C goes straight to it; it ramps the arms
# home before exiting, and a second Ctrl-C skips that ramp) -------------------
echo "[run] starting bimanual bridge — B/Y arms/disarms, Ctrl-C exits"
rc=0
"$ENV_BIN/python" "$PKG/robots/yam_ultra/teleop_bimanual.py" --ws-url "$WS_URL" "$@" || rc=$?
echo "[run] bridge exited (rc=$rc) — cleaning up"
exit "$rc"
