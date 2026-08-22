# yam_ultra

Bimanual [i2rt YAM-Ultra](https://github.com/i2rt-robotics/i2rt): two 6-DoF
arms with LINEAR_4310 parallel-jaw grippers on CAN, three RealSense cameras,
driven either by Quest teleoperation or a LeRobot policy.

```
--robot.type=yam_ultra_bimanual
--teleop.type=bi_quest_teleop | single_arm_quest_teleop
```

```
follower.py         LeRobot Robot adapter — the thing every lerobot-* CLI uses
sim_follower.py     same schema, backed by the Isaac twin over HTTP
arm_server.py       one arm, one process, owns CAN + torque
arm_client.py       talks to an arm_server; duck-types an i2rt robot
connection.py       the single place that calls i2rt's get_yam_robot()
teleop_bimanual.py  standalone both-arms teleop entry point
teleop_single.py    standalone single-arm teleop entry point
ik/                 decoupled 3+3 differential IK
model/              MJCF + meshes + the model the IK compiles
teleop/             LeRobot Teleoperator adapters
config/             rig identity — camera serials, presets, intrinsics
```

## Three processes, not one

```
┌─ lerobot-rollout / lerobot-record ──────────────────┐
│   policy inference (GPU) · cameras · YamUltraFollower│
│     └── YamArmClient ×2                              │
└──────────┬───────────────────────┬───────────────────┘
           │ portal RPC (localhost) │
┌──────────▼──────────┐  ┌──────────▼──────────┐
│ arm_server can_left │  │ arm_server can_right│
│   └ i2rt MotorChain │  │   └ i2rt MotorChain │
└─────────────────────┘  └─────────────────────┘
```

**Why.** i2rt's CAN thread must keep sending inside each motor's firmware
watchdog window. In-process it shared a GIL with policy inference, got
starved, and the motors reported `loss communication` (error `0xD`).

`arm_client.py` deliberately exposes the same four methods the follower used
to call on `MotorChainRobot` — `num_dofs()`, `get_joint_pos()`,
`command_joint_pos()`, `close()` — so all the hard-won follower logic (Δq
clamp, park-on-disconnect, dead-loop detection, gripper flip, camera handling)
still runs unchanged. Only the transport underneath moved.

**The servers own the torque.** They park the arms on Ctrl-C, so killing one
directly is as safe as a clean disconnect. Corollary: don't run
`teleop_demo.sh` while they're up, or two control loops are commanding the
same motors.

**Known cost:** portal's client socket spins rather than blocking while a call
is in flight — roughly a core per arm under load. There is no config knob for
it. It was accepted to keep single-arm operation possible.

## Running

```bash
./scripts/start_arm_servers.sh          # --sim for i2rt SimRobots
```

Default ports **11333** (left) / **11334** (right), channels `can_left` /
`can_right`. Then any LeRobot CLI with `--robot.type=yam_ultra_bimanual`, or:

```bash
./scripts/teleop_demo.sh                # VR teleoperation
./scripts/record.sh                     # teleop dataset recording
./scripts/rollout.sh                    # policy on the real arms
./scripts/rollout.sh --mode=live        # ... retargetable while running
./scripts/harness.sh                    # ... driven by a high-level agent
```

`yam_ultra_sim` is the same schema backed by the Isaac twin — a policy rollout
runs against it with one flag changed and no policy changes. See
[`sim_follower.py`](sim_follower.py) and
[`../../rollout/README.md`](../../rollout/README.md) for `--mode=live`.

## Config worth knowing

| field | default | note |
|---|---|---|
| `max_relative_target` | `[.133,.133,.133,.15,.15,.15]` | per-joint Δq bound for one tick, in rad; scalar broadcasts, `None` disables |
| `park_on_disconnect` | `True` | ramps to all-zeros over `park_duration_s` |
| `gripper_flip` | `False` | invert gripper mapping if your rig is wired the other way |

**`max_relative_target` does not limit velocity.** Nothing bounds how fast the
motors close a commanded step — that's the PD gains. It bounds how large a step
may be per tick, and bounding the step is what bounds the resulting error
spike.

**A slow tick and a slow arm are the same thing.** One tick consumes one action
of a trajectory the policy learned at 30 fps, so a loop that misses 30 Hz plays
it in slow motion. If the arm feels sluggish, measure the loop before touching
this table.

**Park is all zeros**, confirmed against i2rt's own documentation — not
"wherever the arm was at connect", which is a pose that may be mid-air.

## Teleop entry points

`teleop_bimanual.py` and `teleop_single.py` are standalone scripts; `teleop/`
holds the LeRobot `Teleoperator` adapters used by `lerobot-record` and
friends. Both paths share one IK loop.

Cameras are disabled (`cameras={}`) in the standalone teleop scripts on
purpose: the relay already owns the physical RealSense devices for the live VR
view, and **two processes cannot open the same RealSense serial**.

`single_arm_quest_teleop` wraps the bimanual teleop and strips the
`left_`/`right_` prefix for one chosen arm — useful for DAgger interventions
on a single-arm policy while the operator still wears both controllers.

## Sub-package docs

- [`ik/`](ik/README.md) — why the IK is decoupled and what that costs
- [`model/`](model/README.md) — MJCF provenance and the deliberate deviations
- [`teleop/`](teleop/README.md) — the Teleoperator adapters and action schema

## Gotchas

- **`--robot.sim` needs a viewer to see anything.** i2rt's `SimRobot` also
  lacks `move_joints`, which the park path handles by falling back.
- **A policy that yanks toward stow is not a rig bug.** Each chunk opening
  with a jump proportional to `|state|` is regression-to-the-mean from an
  underfit model. `scripts/analysis/inspect_policy_chunks.py` shows it
  directly, and no follower-side clamp will fix it.
- **i2rt is pinned deliberately** at a pre-v1.2.4 commit; v1.2.4 flips the
  Coulomb friction feedforward off by default, which changes arm feel.
