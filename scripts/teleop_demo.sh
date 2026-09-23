#!/usr/bin/env bash
# Launch the YAM-Ultra teleop stack in one terminal: the relay (TLS, reused if
# one is already on :8443), then the bimanual bridge in the foreground.
#
#   ./scripts/teleop_demo.sh [args forwarded to teleop_bimanual.py]
#   ./scripts/teleop_demo.sh --sim              # no hardware dry-run
#   ./scripts/teleop_demo.sh --sim --rerun-poses # inspect XR and EE poses
#   ./scripts/teleop_demo.sh --left-channel can_right --right-channel can_left
#
# Ctrl-C stops the bridge, ramping the arms home first, then tears down the
# relay if this script started it. A second Ctrl-C skips the ramp.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PKG="$ROOT/src/lerobot_robot_sparklab"
WS_URL=wss://127.0.0.1:8443/ws
PYTHON_BIN=${PYTHON_BIN:-/opt/miniforge/envs/robot-py312/bin/python}

# Use the target Conda interpreter and this checkout, regardless of the caller's
# active shell environment.
[[ -x "$PYTHON_BIN" ]] || {
    echo "[run] Python not found: $PYTHON_BIN" >&2; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" - <<'PY' || { echo "[run] dependencies missing from robot-py312" >&2; exit 1; }
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("lerobot_robot_sparklab") else 1)
PY
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
    setsid "$PYTHON_BIN" -m lerobot_robot_sparklab.relay.server --host 0.0.0.0 \
        --cameras-yaml "$PKG/robots/yam_ultra/config/cameras.yaml" \
        --ssl-keyfile "$ROOT/key.pem" --ssl-certfile "$ROOT/cert.pem" \
        > "$LOG_DIR/relay.log" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 1 20); do
        curl -sk --max-time 1 -o /dev/null https://127.0.0.1:8443/ && break
        sleep 0.5
    done
    curl -sk --max-time 2 -o /dev/null https://127.0.0.1:8443/ \
        || { echo "[run] relay failed to start — see $LOG_DIR/relay.log"; exit 1; }
fi
# Prefer a private lab-LAN address: the campus network drops inbound from other
# subnets, so only the lab router reaches the headset. Override with LAN_IP=.
pick_lan_ip() {
    if [ -n "${LAN_IP:-}" ]; then echo "$LAN_IP"; return; fi
    local addrs private
    addrs=$(ip -4 -o addr show scope global \
        | grep -vE ' (docker|veth|br-|virbr)' \
        | awk '{split($4,a,"/"); print a[1]}')
    private=$(echo "$addrs" | grep -E '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)' | head -1)
    if [ -n "$private" ]; then echo "$private"; else echo "$addrs" | head -1; fi
}
LAN_IP=$(pick_lan_ip)
if ! openssl x509 -in "$ROOT/cert.pem" -noout -ext subjectAltName 2>/dev/null \
        | grep -q "IP Address:$LAN_IP"; then
    echo "[run] WARNING: cert.pem does not cover $LAN_IP — the headset will reject it." >&2
    echo "[run]          fix with: ./scripts/make_certs.sh" >&2
fi
echo "[run] Quest browser → https://$LAN_IP:8443/"

# To watch the arms, run Isaac's live viewport separately (works over SSH):
#     ./scripts/isaac_python.sh -m sparklab_sim.live   ->  http://127.0.0.1:8080/

# ---- bridge (foreground; Ctrl-C ramps the arms home, twice skips it) --------
echo "[run] starting bimanual bridge — B/Y arms/disarms, Ctrl-C exits"
rc=0
"$PYTHON_BIN" "$PKG/robots/yam_ultra/teleop_bimanual.py" --ws-url "$WS_URL" "$@" || rc=$?
echo "[run] bridge exited (rc=$rc) — cleaning up"
exit "$rc"
