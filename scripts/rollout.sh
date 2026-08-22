#!/usr/bin/env bash
# Run a policy. Robot, policy and strategy are three independent choices.
#
#     ./scripts/rollout.sh                              # real arms, fine-tuned
#     ./scripts/rollout.sh --policy=stock-typed         # stock baseline
#     ./scripts/rollout.sh --mode=dagger                # teleop takeover
#     ./scripts/rollout.sh --robot=sim --policy=stock   # Isaac twin
#     ./scripts/rollout.sh --mode=live                  # retargetable while running
#     ./scripts/rollout.sh --dry-run ...                # print the command only
#
# Unrecognised flags pass through to the runner, so any lerobot-rollout flag
# works without editing this file:
#
#     ./scripts/rollout.sh --duration=120 --policy.num_flow_timesteps=4
#
# Env knobs: FINETUNED_PATH STOCK_REPO FOXGLOVE_PORT CONTROL_PORT CPU_CORES
#            DAGGER_EPISODES DAGGER_REPO_ID DAGGER_RECORD_AUTONOMOUS
#            RELAY_WS_URL ROLLOUT_SKIP_PREFLIGHT

set -euo pipefail

# Checkpoint paths are relative to the repo root (../SparkRobot/...), so anchor
# there rather than to wherever this was called from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FINETUNED_PATH="${FINETUNED_PATH:-../SparkRobot/MolmoAct2/lerobot_training/last/pretrained_model}"
STOCK_REPO="${STOCK_REPO:-lerobot/MolmoAct2-BimanualYAM-LeRobot}"
FOXGLOVE_PORT="${FOXGLOVE_PORT:-8765}"
CONTROL_PORT="${CONTROL_PORT:-8090}"

ROBOT=hw
POLICY=finetuned
MODE=base
TASK="put the marker into the cardboard box"
DURATION=""          # resolved after parsing; the default depends on the mode
USE_FOXGLOVE=1
DRY_RUN=0
PASSTHRU=()

usage() {
    cat <<'EOF'
usage: rollout.sh [--robot=hw|sim] [--policy=finetuned|stock|stock-typed]
                  [--mode=base|dagger|live] [--task=STR] [--duration=N]
                  [--no-foxglove] [--dry-run] [extra lerobot-rollout flags...]

  --robot    hw          real arms, via the arm_server processes on CAN (default)
             sim         the Isaac twin, via sparklab_sim.policy_server

  --policy   finetuned   local fine-tune at $FINETUNED_PATH (default)
             stock       stock checkpoint as a LeRobot policy dir; gets the
                         camera-key rename its input_features need
             stock-typed stock checkpoint configured by --policy.type, with
                         every setup flag spelled out and no rename. Allows
                         network on the first run so it can download.

  --mode     base        autonomous rollout (default)
             dagger      human-in-the-loop correction capture; hw only, needs
                         the relay up for Quest takeover
             live        control port for retargeting mid-run, driven from
                         `sparklab-rollout-ctl` or harness.sh

env: FINETUNED_PATH STOCK_REPO FOXGLOVE_PORT CONTROL_PORT DAGGER_REPO_ID
     ROLLOUT_SKIP_PREFLIGHT=1  skip the "is it listening" checks
EOF
}

die() { printf 'rollout.sh: %s\n' "$1" >&2; exit 2; }

# --- arguments --------------------------------------------------------------
# Both `--task=x` and `--task x` are accepted. The space-separated form matters:
# without it, `--task put the marker in` falls through to PASSTHRU as five
# separate words and lerobot-rollout sees --task twice.
while [ $# -gt 0 ]; do
    arg="$1"; val="${arg#*=}"
    case "$arg" in
        --robot|--policy|--mode|--task|--duration)
            [ $# -ge 2 ] || die "$arg needs a value"
            val="$2"; shift ;;
    esac
    case "$arg" in
        --robot|--robot=*)       ROBOT="$val" ;;
        --policy|--policy=*)     POLICY="$val" ;;
        --mode|--mode=*)         MODE="$val" ;;
        --task|--task=*)         TASK="$val" ;;
        --duration|--duration=*) DURATION="$val" ;;
        --no-foxglove)           USE_FOXGLOVE=0 ;;
        --dry-run)               DRY_RUN=1 ;;
        -h|--help)               usage; exit 0 ;;
        *)                       PASSTHRU+=("$arg") ;;
    esac
    shift
done

case "$ROBOT"  in hw|sim) ;;                      *) die "unknown --robot=$ROBOT (hw|sim)" ;; esac
case "$POLICY" in finetuned|stock|stock-typed) ;; *) die "unknown --policy=$POLICY (finetuned|stock|stock-typed)" ;; esac
case "$MODE"   in base|dagger|live) ;;            *) die "unknown --mode=$MODE (base|dagger|live)" ;; esac

# dagger needs a human taking over through the Quest relay onto real arms.
if [ "$MODE" = dagger ] && [ "$ROBOT" != hw ]; then
    die "--mode=dagger requires --robot=hw"
fi

# dagger is bounded by --strategy.num_episodes, and a wall-clock limit would cut
# a session off mid-correction. lerobot-rollout treats 0 as infinite.
if [ -z "$DURATION" ]; then
    case "$MODE" in dagger) DURATION=0 ;; *) DURATION=2000 ;; esac
fi

# --- preflight --------------------------------------------------------------
# A missing dependency otherwise surfaces as a long opaque timeout inside
# connect(), after the ~90 s model load. Check the ports up front instead.

port_open() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | grep -qE "[:.]$1[[:space:]]"
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3<&- 3>&-
    fi
}

# require_port <port>[,<port>...] <what> <how-to-start>
require_port() {
    if [ "${ROLLOUT_SKIP_PREFLIGHT:-0}" = "1" ]; then return 0; fi
    local ports="$1" what="$2" how="$3" missing=()
    local IFS=','
    for p in $ports; do port_open "$p" || missing+=("$p"); done
    unset IFS
    if [ ${#missing[@]} -eq 0 ]; then
        printf '[preflight] %-14s %-12s ... OK\n' "$what" "$ports" >&2
        return 0
    fi
    printf '[preflight] %-14s %-12s ... NOT LISTENING (%s)\n' "$what" "$ports" "${missing[*]}" >&2
    printf '            start it with: %s\n' "$how" >&2
    printf '            (ROLLOUT_SKIP_PREFLIGHT=1 to bypass)\n' >&2
    exit 1
}

# --- robot ------------------------------------------------------------------
ROBOT_ARGS=()
PIN=()

if [ "$ROBOT" = hw ]; then
    require_port "11333,11334" "arm servers" "./scripts/start_arm_servers.sh"
    if [ "$MODE" = dagger ]; then
        require_port "8443" "relay" \
            "sparklab-relay --host 0.0.0.0 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem"
    fi

    ROBOT_ARGS=(
        --robot.type=yam_ultra_bimanual
        --robot.left_channel=can_left
        --robot.right_channel=can_right
    )

    cat >&2 <<'EOF'
[hw] The arm servers own the CAN loop and the torque. Do not run teleop_demo.sh
     alongside them — that is two control loops on the same motors.
[hw] Auto-recovery is a SERVER flag (arm_server --enable-auto-recovery), not a
     --robot.* one. Leaving it off is the better default: it re-asserts the
     pre-error target at full gains from inside i2rt's driver, which the
     follower's velocity clamp cannot bound.
EOF

    # The ~21 GB base model is already cached, but huggingface_hub still makes a
    # network round trip per file to check etags before falling through to it.
    # That is most of what "Fetching N files ... 0.00B/0.00B" costs, and it
    # starves i2rt's CAN polling threads right after connect().
    export HF_HUB_OFFLINE=1

    # CPU pin, OFF by default. Pinning starved the loop it was meant to protect:
    # on this 24-CPU workstation a live rollout is ~400% CPU across 140 threads
    # and the two arm servers another ~171%, against the 600% ceiling that
    # `taskset -c 0-5` imposes. At that saturation the loop misses its deadline,
    # and since one tick consumes one action of a 30 fps trajectory, a missed
    # deadline is a slow arm — 30 Hz unpinned vs ~4 Hz pinned, same checkpoint.
    # Set CPU_CORES=0-5 to restore it if you have a specific neighbour to
    # isolate from. Note it was never intra-process isolation anyway: inference
    # and CAN polling share one GIL, which affinity does not change.
    if [ -n "${CPU_CORES-}" ]; then PIN=( taskset -c "$CPU_CORES" ); fi
else
    require_port "8081" "sim policy" \
        "./scripts/isaac_python.sh -m sparklab_sim.policy_server --port 8081 --physics"
    ROBOT_ARGS=( --robot.type=yam_ultra_sim )
fi

# --- policy -----------------------------------------------------------------
# The camera-key rename maps the checkpoint's top/left/right onto the rig's
# top/left_wrist/right_wrist. It keys off the POLICY, not the robot: it is the
# checkpoint's input_features that disagree, on either robot.
RENAME=(
    --rename_map='{"observation.images.top": "observation.images.top", "observation.images.right": "observation.images.right_wrist", "observation.images.left": "observation.images.left_wrist"}'
)

POLICY_ARGS=()
RENAME_ARGS=()

case "$POLICY" in
    # A --policy.path load gets NOTHING added: the checkpoint's config.json is
    # the source of truth for dtype, chunking, cuda graphs and amp. In
    # particular do not add --policy.use_amp=true — every checkpoint we load
    # says use_amp: false, and the engine reads that flag as
    # `torch.autocast(device_type="cuda")`, which defaults to FLOAT16. Forcing
    # it wraps an already-bfloat16 model in fp16 autocast: 5 exponent bits
    # instead of 8 for a flow head integrating 8 steps, plus a bf16->fp16->bf16
    # cast around every eligible op. Slower and worse. Measured on hardware.
    finetuned)
        POLICY_ARGS=( --policy.path="${FINETUNED_PATH}" )
        RENAME_ARGS=( "${RENAME[@]}" )
        ;;
    stock)
        POLICY_ARGS=( --policy.path="${STOCK_REPO}" )
        RENAME_ARGS=( "${RENAME[@]}" )
        ;;
    # Configured by --policy.type, so nothing is inferred from a config.json and
    # every setup flag is spelled out. --policy.image_keys names the wrist keys
    # directly, so this variant must NOT also get the rename.
    stock-typed)
        POLICY_ARGS=(
            --policy.type=molmoact2
            --policy.checkpoint_path="${STOCK_REPO}"
            --policy.action_mode=continuous
            --policy.setup_type="bimanual yam robotic arms in molmoact2"
            --policy.control_mode="absolute joint pose"
            --policy.normalize_gripper=false
            --policy.num_flow_timesteps=8
            --policy.image_keys='["observation.images.top","observation.images.left_wrist","observation.images.right_wrist"]'
            --policy.inference_action_mode=continuous
            --policy.device=cuda
            --policy.model_dtype=bfloat16
            --policy.enable_inference_cuda_graph=true
            --policy.n_action_steps=30
            --policy.chunk_size=30
        )
        # This is the baseline you reach for precisely when it may not be cached
        # yet, and offline turns a download into "not found in cache". Cached
        # runs lose only the per-file etag round trips.
        unset HF_HUB_OFFLINE
        ;;
esac

# --- strategy ---------------------------------------------------------------
STRATEGY_ARGS=( --strategy.type=base )

if [ "$MODE" = dagger ]; then
    STRATEGY_ARGS=(
        --strategy.type=dagger
        --strategy.input_device=keyboard
        # false records ONLY the correction windows, one episode each — the most
        # information-dense choice for filling the recovery gap. true is
        # sentry-style continuous capture: more volume, less signal.
        --strategy.record_autonomous="${DAGGER_RECORD_AUTONOMOUS:-false}"
        --strategy.num_episodes="${DAGGER_EPISODES:-20}"
        --teleop.type=bi_quest_teleop
        --teleop.ws_url="${RELAY_WS_URL:-wss://127.0.0.1:8443/ws}"
        # A new repo_id, NOT put-box: DAgger tags every frame with an
        # `intervention` bool that put-box's schema lacks, so this cannot be
        # appended without a schema migration first.
        --dataset.repo_id="${DAGGER_REPO_ID:-minhphd/put-box-dagger}"
        --dataset.single_task="${TASK}"
        --dataset.fps=30
        --dataset.push_to_hub=false
    )
    cat >&2 <<'EOF'
[dagger] keyboard: p pause/resume | c toggle correction recording (while
         paused) | u push to hub | ESC stop. No foot pedal attached, so this
         needs a second person at the keyboard while you are in the headset.
EOF
fi

# --- runner -----------------------------------------------------------------
if [ "$MODE" = live ]; then
    # A different runner: it wraps the rollout in a control server so the task
    # can be retargeted mid-run instead of paying the ~90 s model load again.
    # That control port is what harness.sh and sparklab-rollout-ctl attach to.
    # Invoked as `python -m` rather than the console script so a stale editable
    # install cannot break the rollout; PYTHONPATH=src below is all it needs.
    export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1
    CMD=( "${PIN[@]}" python -m lerobot_robot_sparklab.rollout.live
          --control-port "$CONTROL_PORT"
          --publish-every "${PUBLISH_EVERY:-5}" )
    # Smoothing and rate are sim-tuned. On hardware, leave LeRobot's own
    # defaults (fps 30) and pass --fps through if you want otherwise.
    if [ "$ROBOT" = sim ]; then
        CMD+=( --interpolation_multiplier 2 --fps 20 )
    fi
    printf '[live] control port %s — attach ./scripts/harness.sh, or a REPL\n' "$CONTROL_PORT" >&2
    printf '[live] with: sparklab-rollout-ctl\n' >&2
else
    CMD=( "${PIN[@]}" lerobot-rollout )
fi

# Foxglove ABORTS on a bound port rather than falling back, so a second
# concurrent rollout dies at startup unless you move FOXGLOVE_PORT.
FOXGLOVE_ARGS=()
if [ "$USE_FOXGLOVE" = 1 ]; then
    FOXGLOVE_ARGS=(
        --display_data true
        --display_mode=foxglove
        --display_ip=127.0.0.1
        --display_port="$FOXGLOVE_PORT"
    )
fi

CMD+=( "${STRATEGY_ARGS[@]}" "${POLICY_ARGS[@]}" "${ROBOT_ARGS[@]}"
       --task="${TASK}" "${RENAME_ARGS[@]}" "${FOXGLOVE_ARGS[@]}"
       --duration="${DURATION}" "${PASSTHRU[@]}" )

printf '[rollout] robot=%s policy=%s mode=%s duration=%s\n' \
    "$ROBOT" "$POLICY" "$MODE" "$DURATION" >&2

if [ "$DRY_RUN" = 1 ]; then
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi
exec "${CMD[@]}"
