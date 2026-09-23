#!/usr/bin/env bash
# Start one arm_server process per arm. Run BEFORE record.sh / rollout.sh /
# anything using --robot.type=yam_ultra_bimanual on real hardware — the follower
# connects to these servers, it does not start them. See DESIGN.md for why the
# arms live in their own processes.
#
# These processes OWN TORQUE. They park the arms home on Ctrl-C / SIGTERM, so
# stopping them is safe — but leave them running between runs, since each start
# re-homes the gripper and recaptures the home pose.
#
#   ./scripts/start_arm_servers.sh              # real hardware, foreground
#   ./scripts/start_arm_servers.sh --sim        # i2rt SimRobots + MuJoCo windows
#   ./scripts/start_arm_servers.sh --sim --no-viewer  # headless simulation
#
# Ctrl-C stops both (parking first). Logs: /tmp/yam-arm-servers/

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-/opt/miniforge/envs/robot-py312/bin/python}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[arm-servers] Python not found: $PYTHON_BIN" >&2
    echo "[arm-servers] set PYTHON_BIN to the robot-py312 interpreter" >&2
    exit 1
fi

# Use this checkout even when the package is not installed editable in the env.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PYTHON_BIN" -c 'import portal; import lerobot_robot_sparklab' 2>/dev/null; then
    echo "[arm-servers] robot dependencies are unavailable in $PYTHON_BIN" >&2
    exit 1
fi

LEFT_CHANNEL=${LEFT_CHANNEL:-can_left}
RIGHT_CHANNEL=${RIGHT_CHANNEL:-can_right}
# Must match YamUltraFollowerConfig.left/right_server_port.
LEFT_PORT=${LEFT_PORT:-11333}
RIGHT_PORT=${RIGHT_PORT:-11334}

EXTRA=()
SIM=false
VIEWER=true
for arg in "$@"; do
    case "$arg" in
        --sim) SIM=true; EXTRA+=("$arg") ;;
        --no-viewer) VIEWER=false ;;
        *) EXTRA+=("$arg") ;;
    esac
done
if $SIM && $VIEWER; then
    EXTRA+=(--viewer)
fi

LOG_DIR=/tmp/yam-arm-servers
mkdir -p "$LOG_DIR"

PIDS=()
CLEANED_UP=false
cleanup() {
    $CLEANED_UP && return
    CLEANED_UP=true
    echo
    echo "[arm-servers] stopping (each parks its arm home first) ..."
    for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
    echo "[arm-servers] stopped."
}
on_signal() {
    cleanup
    exit 0
}
trap on_signal INT TERM
trap cleanup EXIT

cd "$ROOT"
for spec in "$LEFT_CHANNEL:$LEFT_PORT" "$RIGHT_CHANNEL:$RIGHT_PORT"; do
    channel=${spec%%:*}
    port=${spec##*:}
    echo "[arm-servers] starting $channel on port $port (log: $LOG_DIR/$channel.log)"
    "$PYTHON_BIN" -m lerobot_robot_sparklab.robots.yam_ultra.arm_server \
        --channel "$channel" --port "$port" "${EXTRA[@]}" \
        > "$LOG_DIR/$channel.log" 2>&1 &
    PIDS+=("$!")
done

port_is_listening() {
    "$PYTHON_BIN" -c \
        'import socket, sys; s = socket.socket(); s.settimeout(0.2); sys.exit(s.connect_ex(("127.0.0.1", int(sys.argv[1]))))' \
        "$1" 2>/dev/null
}

startup_failed=false
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    port=${LEFT_PORT}
    [[ $i -eq 1 ]] && port=${RIGHT_PORT}
    ready=false
    for _ in $(seq 1 150); do
        state=$(ps -o stat= -p "$pid" 2>/dev/null || true)
        if ! kill -0 "$pid" 2>/dev/null || [[ -z "$state" || "$state" == Z* ]]; then
            break
        fi
        if port_is_listening "$port"; then
            ready=true
            break
        fi
        sleep 0.1
    done
    if ! $ready; then
        startup_failed=true
    fi
done
if $startup_failed; then
    echo "[arm-servers] a server died on startup; recent logs:" >&2
    tail -n 20 -- "$LOG_DIR"/*.log >&2 || true
    exit 1
fi

echo "[arm-servers] both up. Leave this running; start record.sh / rollout.sh"
echo "[arm-servers] in another terminal. Ctrl-C here parks both arms and exits."
wait -n "${PIDS[@]}"
rc=$?
echo "[arm-servers] a server exited unexpectedly (rc=$rc); recent logs:" >&2
tail -n 20 -- "$LOG_DIR"/*.log >&2 || true
exit 1
