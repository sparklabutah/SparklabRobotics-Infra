# sparklab-robotics

SparkLab's robotics stack: Meta Quest VR teleoperation, a camera/pose relay,
[LeRobot](https://github.com/huggingface/lerobot) `Robot`/`Teleoperator`
plugins for the lab's robots, an Isaac digital twin, and a web harness for
putting a high-level agent in front of a low-level VLA. Shared infrastructure
lives at the top level; each robot is a self-contained subpackage.

First robot: the bimanual [i2rt YAM-Ultra](https://github.com/i2rt-robotics/i2rt).

## Layout

```
src/lerobot_robot_sparklab/          the LeRobot plugin distribution
├── relay/      FastAPI WebSocket relay + WebRTC camera streams + the WebXR
│               page served to the Quest. Robot-agnostic.
├── quest/      Robot-agnostic Quest input: xr_frame subscription, staleness,
│               xr-standard button mapping.
├── core/       Clutch-relative pose mapping with absorbing reach limits.
├── cameras/    RealSense tooling (calibration, presets, intrinsics export).
├── harness/    Web UI + REST for driving the VLA from a high-level agent.
├── rollout/    A rollout plus a control port, and a REPL to drive it. Runs on
│               hardware or sim — it is the robot that differs, not this.
├── tools/      No-hardware dry-run/inspection utilities.
└── robots/
    └── yam_ultra/   kinematics, IK, hardware bridge, teleoperator, rig config

src/sparklab_sim/                    the Isaac twin — its own interpreter
                 scene building, rendering, and a policy server the rollout
                 talks to over HTTP

scripts/         reusable workflows (rollout, record, teleop, health, analysis)
```

The split is deliberate: **anything that would have to change for a different
arm belongs under `robots/`.** If shared code needs a robot-specific tweak,
that's a signal something leaked.

`sparklab_sim` is a separate package because Isaac ships its own Python with
no lerobot; the two interpreters cannot be merged, so they talk over a socket.
That makes the boundary mechanical rather than a matter of taste: **a module
that imports lerobot cannot live in `sparklab_sim`,** whatever it is about.
"Sim vs real" is the wrong axis — `rollout/` drives both robots and belongs on
the lerobot side; `yam_ultra_sim` is a lerobot plugin for a simulated arm and
belongs under `robots/`.

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

The arm servers own the CAN loop and the torque. Start them first, leave them
running; everything else is a client of them.

```bash
./scripts/start_arm_servers.sh          # --sim for i2rt SimRobots

# then one of:
./scripts/teleop_demo.sh                # VR teleoperation
./scripts/record.sh                     # teleop dataset recording
./scripts/rollout.sh                    # policy on the real arms
./scripts/rollout.sh --mode=live        # ... plus a control port
./scripts/harness.sh                    # ... plus an agent + web UI on :8099
```

`./scripts/rollout.sh --dry-run` prints the command it would run without
running it — the fastest way to see what a flag actually does.

## Scripts

| script | purpose |
|---|---|
| `start_arm_servers.sh` | one `arm_server` per arm; owns CAN + torque, parks on Ctrl-C |
| `rollout.sh` | **the policy entry point.** `--robot=hw\|sim`, `--policy=finetuned\|stock\|stock-typed`, `--mode=base\|dagger\|live` |
| `harness.sh` | web UI + high-level agent on top of `--mode=live` |
| `teleop_demo.sh` | relay + bimanual VR bridge in one terminal |
| `record.sh` | teleop dataset recording via `lerobot-record` |
| `can_health.sh` | SocketCAN fault counters — separates a wire problem from a host-timing one |
| `analysis/profile_rollout.py` | per-stage timing of a real tick — measures inference instead of inferring it |
| `analysis/validate_policy_pipeline.py` | run a checkpoint open-loop on its own training data — catches undertraining and pipeline bugs *before* powering the arms |
| `analysis/inspect_policy_chunks.py` | dump unclamped action chunks; detects a policy that yanks toward stow |
| `analysis/lang_probe.py` | does the policy read the instruction, or ignore it? |

Every script is self-contained — no shared library to jump to. Details and the
full flag matrix in [`scripts/README.md`](scripts/README.md).

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

Measured on this rig, not assumed. Kept here so they aren't rediscovered.

- **A checkpoint's `config.json` is the source of truth for inference.** Do
  not re-specify dtype, chunking or amp from a script. Forcing
  `--policy.use_amp=true` onto a checkpoint that says `use_amp: false` wraps a
  bfloat16 model in fp16 autocast — that flag *is*
  `torch.autocast(device_type="cuda")`, which defaults to **float16** — giving
  narrower range (max 65504 vs 3.4e38) and a cast around every op. Slower *and*
  worse actions. `rollout.sh` adds nothing to a `--policy.path` load.

- **Loop rate is arm speed.** One tick consumes one action, and a chunk is a
  trajectory the policy learned at 30 fps. A loop that misses 30 Hz plays that
  trajectory in slow motion — this is not a velocity setting. So anything
  costing per-tick time (foxglove `--display_data`, a tight `taskset`) shows
  up as a visibly slower arm. `rollout/live.py` logs `loop running slower
  than target` when it happens.

- **The sync engine preprocesses every tick and discards 29 of 30.** Measured
  12.47 ms of image resize + tokenisation per pop-only tick, whose output
  `select_action` never reads. Skipping it was tried and **reverted** — it made
  the policy visibly worse, so the redundant work is load-bearing somewhere.
  Do not re-add a preprocessor bypass without measuring behaviour, not just
  tick time.

- **`max_relative_target` is a position-*step* cap, not a velocity limit.**
  Nothing bounds how fast the motors close a commanded step — that's the PD
  gains. Bounding the step is what bounds the resulting error spike, which is
  why the step is what gets capped.

- **RTC inference needs `n_action_steps / fps > inference_time`**, or it
  discards whole chunks and the arm doesn't move. `--inference.type=sync`
  instead stalls the loop for the whole inference — ~275 ms measured, once
  every 30 ticks.

- **Validate checkpoints before hardware.** A policy whose error exceeds the
  motion it must command will move erratically no matter how healthy the rig
  is; `scripts/analysis/validate_policy_pipeline.py` catches that in minutes
  without powering anything.

## Adding a robot

1. `mkdir src/lerobot_robot_sparklab/robots/<name>/`
2. Write a `Robot` subclass and register it:
   `@RobotConfig.register_subclass("<name>")`
3. Re-export it from `robots/<name>/__init__.py` — **required**, see below.
4. Add `from .robots import <name>` to the package `__init__.py`.

Then `--robot.type=<name>` works in every `lerobot-*` CLI. Reuse `quest/` +
`core/` for teleoperation and write only the mapping from controller poses to
your robot's action space. Details in
[`robots/README.md`](src/lerobot_robot_sparklab/robots/README.md).

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

## License

Apache-2.0.
