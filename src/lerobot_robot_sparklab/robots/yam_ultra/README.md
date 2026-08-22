# yam_ultra

Bimanual [i2rt YAM-Ultra](https://github.com/i2rt-robotics/i2rt): two 6-DoF
arms with LINEAR_4310 parallel-jaw grippers on CAN, three RealSense cameras,
driven either by Quest teleoperation or a LeRobot policy.

```
--robot.type=yam_ultra_bimanual | yam_ultra_sim
--teleop.type=bi_quest_teleop | single_arm_quest_teleop
```

```
follower.py         LeRobot Robot adapter — what every lerobot-* CLI uses
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
┌─ lerobot-rollout / lerobot-record ───────────────────┐
│   policy inference (GPU) · cameras · YamUltraFollower │
│     └── YamArmClient ×2                               │
└──────────┬────────────────────────┬───────────────────┘
           │ portal RPC (localhost) │
┌──────────▼──────────┐  ┌──────────▼───────────┐
│ arm_server can_left │  │ arm_server can_right │
│   └ i2rt MotorChain │  │   └ i2rt MotorChain  │
└─────────────────────┘  └──────────────────────┘
```

`arm_client.py` exposes the same four methods the follower used to call on
`MotorChainRobot` — `num_dofs()`, `get_joint_pos()`, `command_joint_pos()`,
`close()` — so the follower's logic runs unchanged and only the transport
moved. Why the split exists, and its cost, are in
[`DESIGN.md`](../../../../DESIGN.md#the-arms-run-in-their-own-processes).

**The servers own the torque.** They park the arms on Ctrl-C, so killing one
directly is as safe as a clean disconnect. Do not run `teleop_demo.sh` while
they are up, or two control loops command the same motors.

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

`yam_ultra_sim` is the same schema backed by the Isaac twin, so a policy
rollout runs against it with one flag changed.

## Config worth knowing

| field | default | note |
|---|---|---|
| `max_relative_target` | `[.133,.133,.133,.15,.15,.15]` | per-joint Δq bound for one tick, in rad; scalar broadcasts, `None` disables |
| `park_on_disconnect` | `True` | ramps to all-zeros over `park_duration_s` |
| `gripper_flip` | `False` | invert gripper mapping if your rig is wired the other way |

`max_relative_target` bounds the size of a per-tick step, not velocity. Park is
all zeros — the folded rest pose — not wherever the arm was at connect. Both,
and why a slow tick reads as a slow arm, are in
[`DESIGN.md`](../../../../DESIGN.md#measured-constraints).

## Teleop entry points

`teleop_bimanual.py` and `teleop_single.py` are standalone scripts; `teleop/`
holds the LeRobot `Teleoperator` adapters used by `lerobot-record`. Both paths
share one IK loop.

Cameras are disabled (`cameras={}`) in the standalone scripts: the relay
already owns the physical RealSense devices for the live VR view, and two
processes cannot open the same RealSense serial.

`single_arm_quest_teleop` wraps the bimanual teleop and strips the
`left_`/`right_` prefix for one chosen arm — for DAgger interventions on a
single-arm policy while the operator wears both controllers.

## Sub-package docs

- [`ik/`](ik/README.md) — the decoupled solver and its tunables
- [`model/`](model/README.md) — MJCF provenance and the deliberate deviations
- [`teleop/`](teleop/README.md) — the Teleoperator adapters and action schema
- [`config/`](config/README.md) — this rig's camera identity

## Known behaviour

- **`--robot.sim` needs a viewer to see anything.** i2rt's `SimRobot` also
  lacks `move_joints`, which the park path falls back around.
- **A policy that yanks toward stow is not a rig bug.** Each chunk opening with
  a jump proportional to `|state|` is regression-to-the-mean from an underfit
  model; `scripts/analysis/inspect_policy_chunks.py` shows it directly, and no
  follower-side clamp will fix it.
- **i2rt is pinned deliberately** at a pre-v1.2.4 commit: v1.2.4 flips the
  Coulomb friction feedforward off by default, which changes arm feel.
