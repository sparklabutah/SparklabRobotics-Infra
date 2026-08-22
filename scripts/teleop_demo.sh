#!/usr/bin/env bash
# Launch the YAM-Ultra teleop stack in one terminal:
#   1. relay (TLS, reused if one is already listening on :8443)
#   2. the bimanual hardware bridge (foreground; its log is your console)
#
# Runs in whatever environment is active — activate the one holding this
# package first (conda activate robot-py312); the check below fails fast if not.
#
# Usage:
#   ./scripts/teleop_demo.sh [args forwarded to teleop_bimanual.py]
#   ./scripts/teleop_demo.sh --sim              # no hardware dry-run
#   ./scripts/teleop_demo.sh --left-channel can_right --right-channel can_left
#
# Ctrl-C stops the bridge (it ramps the arms home first), then tears down the
# relay if this script started it. A second Ctrl-C skips the ramp.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PKG="$ROOT/src/lerobot_robot_sparklab"
WS_URL=wss://127.0.0.1:8443/ws

# Fail fast on the wrong env rather than deep inside an import.
python - <<'PY' || { echo "[run] activate the env holding this package: conda activate robot-py312" >&2; exit 1; }
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("lerobot_robot_sparklab") else 1)
PY
command -v sparklab-relay >/dev/null || {
    echo "[run] sparklab-relay not on PATH — pip install -e . in the active env" >&2; exit 1; }
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
        sparklab-relay --host 0.0.0.0 \
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
# Which address to point the headset at. Prefer a PRIVATE lab-LAN address over
# the campus one: since 2026-08-12 the campus network drops inbound from other
# subnets, so a headset on WiFi cannot reach the campus address at all -- the
# lab router is the only path that works. `hostname -I` ordering is not stable
# enough to rely on. Override with LAN_IP=... if the guess is wrong.
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

# ---- bridge (foreground — Ctrl-C goes straight to it; it ramps the arms
# home before exiting, and a second Ctrl-C skips that ramp) -------------------
echo "[run] starting bimanual bridge — B/Y arms/disarms, Ctrl-C exits"
rc=0
python "$PKG/robots/yam_ultra/teleop_bimanual.py" --ws-url "$WS_URL" "$@" || rc=$?
echo "[run] bridge exited (rc=$rc) — cleaning up"
exit "$rc"
