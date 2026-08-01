# sparklab-robotics

SparkLab's robotics stack: Meta Quest VR teleoperation, a camera/pose relay,
and [LeRobot](https://github.com/huggingface/lerobot) `Robot`/`Teleoperator`
plugins for the lab's robots. Shared infrastructure lives at the top level;
each robot is a self-contained subpackage.

First robot: the bimanual [i2rt YAM-Ultra](https://github.com/i2rt-robotics/i2rt).

## Layout

```
src/lerobot_robot_sparklab/
├── relay/      FastAPI WebSocket relay + WebRTC camera streams + the WebXR
│               page served to the Quest. Robot-agnostic.
├── quest/      Robot-agnostic Quest input: xr_frame subscription, staleness,
│               xr-standard button mapping.
├── core/       Clutch-relative pose mapping with absorbing reach limits.
├── cameras/    RealSense tooling (calibration, presets, intrinsics export).
├── tools/      No-hardware dry-run/inspection utilities.
└── robots/
    └── yam_ultra/   kinematics, IK, hardware bridge, teleoperator, rig config
scripts/        Reusable LeRobot workflows (record, rollout, validate, health)
```

The split is deliberate: **anything that would have to change for a different
arm belongs under `robots/`.** If shared code needs a robot-specific tweak,
that's a signal something leaked.

## Adding a robot

1. `mkdir src/lerobot_robot_sparklab/robots/<name>/`
2. Write a `Robot` subclass and register it:
   `@RobotConfig.register_subclass("<name>")`
3. Re-export it from `robots/<name>/__init__.py` — **required**, see below.
4. Add `from .robots import <name>` to the package `__init__.py`.

Then `--robot.type=<name>` works in every `lerobot-*` CLI.

Reuse `quest/` + `core/` for teleoperation and write only the mapping from
controller poses to your robot's action space.

### Two naming rules that are load-bearing

**The distribution name must keep the `lerobot_robot_` prefix.** Every
`lerobot-*` CLI calls `register_third_party_plugins()`, which scans installed
*distributions* for that prefix and imports the matching top-level module.
That import runs the registration decorators, so all SparkLab robots become
valid `--robot.type=` choices with no wrapper script.

**Each robot must re-export its own Robot + Config from its subpackage
`__init__.py`.** Once a config is chosen, LeRobot's
`make_device_from_device_class` looks for the Robot class only in the
**direct parent package** of the config's module — it never walks up to the
package root. Miss this and you get
`ImportError: Could not locate device class`.

## Install

```bash
conda create -n sparklab python=3.12 && conda activate sparklab
pip install -e ".[relay,realsense,yam-ultra]"

# i2rt (YAM arm driver — not on PyPI):
git clone https://github.com/i2rt-robotics/i2rt && pip install -e ./i2rt

# torchcodec (lerobot's video decoder) needs a newer libstdc++ than the
# system one; conda-forge ffmpeg + activating the env supplies it:
conda install -c conda-forge ffmpeg
```

`lerobot>=0.6` requires Python ≥3.12. Always `conda activate` rather than
calling the interpreter by path — the activation hook sets `LD_LIBRARY_PATH`,
without which `torchcodec` fails to load.

## Quick start (YAM-Ultra)

```bash
# 1. arm servers own the CAN loop and the torque — start first, leave running
./scripts/start_arm_servers.sh          # --sim for i2rt SimRobots

# 2. relay (Quest pose + camera streams)
sparklab-relay --host 0.0.0.0 \
    --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem

# 3. teleop, recording, or policy rollout
./scripts/teleop_demo.sh
./scripts/record.sh
./scripts/rollout.sh
```

## Scripts

| script | purpose |
|---|---|
| `start_arm_servers.sh` | one `arm_server` per arm; owns CAN + torque, parks on Ctrl-C |
| `teleop_demo.sh` | relay + MuJoCo viewers + bimanual bridge in one terminal |
| `record.sh` | teleop dataset recording via `lerobot-record` |
| `rollout.sh` / `rollout_pretrained.sh` / `rollout_dagger.sh` | policy rollout: fine-tuned / baseline / human-in-the-loop |
| `validate_policy_pipeline.py` | run a checkpoint open-loop on its own training data — catches undertraining and pipeline bugs *before* powering the arms |
| `inspect_policy_chunks.py` | dump unclamped action chunks; detects a policy that yanks toward stow |
| `can_health.sh` | SocketCAN fault counters — separates a wire problem from a host-timing one |

## Why the arms run in their own processes

i2rt's CAN thread has to keep sending inside each motor's watchdog window.
In-process it shared a GIL with policy inference, got starved, and the motors
reported `loss communication` — while every SocketCAN fault counter read
zero, i.e. the wire was clean and frames simply weren't sent in time. So each
arm runs as an `arm_server` process and the follower talks to it over portal
RPC. `scripts/can_health.sh` is what distinguishes those two causes.

Those servers **own the torque**: they park the arms on Ctrl-C, and killing
one directly is as safe as a clean disconnect. Don't run `teleop_demo.sh`
while they're up — you'd have two control loops on the same motors.

## Known constraints

- **`max_joint_velocity` is a position-*step* cap, not a velocity limit.**
  Nothing bounds how fast the motors close a commanded step — that's the PD
  gains. Below `1/max_tick_s` (10 Hz default) achievable speed also falls off
  linearly.
- **RTC inference needs `n_action_steps / fps > inference_time`**, or it
  discards whole chunks and the arm doesn't move. `--inference.type=sync`
  instead stalls the loop on every tick.
- **Validate checkpoints before hardware.** A policy whose error exceeds the
  motion it must command will move erratically no matter how healthy the rig
  is; `validate_policy_pipeline.py` catches that in minutes without powering
  anything.

## License

Apache-2.0.
