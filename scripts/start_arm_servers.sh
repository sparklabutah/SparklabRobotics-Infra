#!/usr/bin/env bash
# Start one arm_server process per arm. Run this BEFORE record.sh /
# rollout.sh / anything that uses --robot.type=yam_ultra_bimanual on real
# hardware — the follower connects to these servers, it does not start them.
#
# Why the arms live in their own processes: i2rt's CAN polling thread has to
# keep sending inside each motor's watchdog window, and in-process it shared
# a GIL with policy inference and checkpoint loading, which starved it into
# `loss communication` while every SocketCAN fault counter read zero (see
# scripts/can_health.sh and robots/yam_ultra/arm_server.py).
#
# These processes OWN TORQUE. They park the arms home on Ctrl-C / SIGTERM,
# so stopping them is safe — but leave them running between runs rather
# than restarting per run: each start re-homes the gripper and recaptures
# the home pose.
#
#   ./scripts/start_arm_servers.sh              # real hardware, foreground
#   ./scripts/start_arm_servers.sh --sim        # i2rt SimRobots, nothing moves
#
# Ctrl-C stops both (parking first). Logs: /tmp/yam-arm-servers/
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"

LEFT_CHANNEL=${LEFT_CHANNEL:-can_left}
RIGHT_CHANNEL=${RIGHT_CHANNEL:-can_right}
# Must match YamUltraFollowerConfig.left/right_server_port.
LEFT_PORT=${LEFT_PORT:-11333}
RIGHT_PORT=${RIGHT_PORT:-11334}

EXTRA=()
for arg in "$@"; do EXTRA+=("$arg"); done

LOG_DIR=/tmp/yam-arm-servers
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
    echo
    echo "[arm-servers] stopping (each parks its arm home first) ..."
    for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
    echo "[arm-servers] stopped."
}
trap cleanup INT TERM EXIT

cd "$ROOT"
for spec in "$LEFT_CHANNEL:$LEFT_PORT" "$RIGHT_CHANNEL:$RIGHT_PORT"; do
    channel=${spec%%:*}
    port=${spec##*:}
    echo "[arm-servers] starting $channel on port $port (log: $LOG_DIR/$channel.log)"
    python -m lerobot_robot_sparklab.robots.yam_ultra.arm_server \
        --channel "$channel" --port "$port" "${EXTRA[@]}" \
        > "$LOG_DIR/$channel.log" 2>&1 &
    PIDS+=($!)
done

sleep 1
for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[arm-servers] a server died on startup — see $LOG_DIR/*.log" >&2
        tail -20 "$LOG_DIR"/*.log >&2 || true
        exit 1
    fi
done

echo "[arm-servers] both up. Leave this running; start record.sh / rollout.sh"
echo "[arm-servers] in another terminal. Ctrl-C here parks both arms and exits."
wait
