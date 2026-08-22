# teleop

LeRobot `Teleoperator` adapters for the YAM-Ultra. Importing these registers
their config types, after which `--teleop.type=` works in any LeRobot CLI.

```
bi_quest_teleop.py          BiQuestTeleoperator — both arms, one IK loop
single_arm_quest_teleop.py  one-arm adapter wrapping the bimanual one
cli.py                      shared entry-point helpers
```

Requires `lerobot` installed.

## Where the robot-specific part starts

The generic half lives upstairs and is reused unchanged:

```
quest/  → controller poses, buttons, staleness   (any robot)
core/   → clutch-relative pose mapping           (any robot)
─────────────────────────────────────────────────
here    → per-arm IK, joint limits, action keys  (YAM-Ultra only)
```

So this package is the mapping from "operator moved their hand this much" to
"these joint targets", and nothing more. A second robot rewrites only this
layer.

## `BiQuestTeleoperator`

Subscribes to the relay's `/ws`, reads `xr_frame` broadcasts, and maintains
**one `ClutchPoseMapper` and one `DecoupledIKSolver` per arm**. On the rising
edge of grip it captures the engage frame and starts running differential IK
each tick.

`get_action()` returns the dict matching `YamUltraFollower.action_features`:

```
{left,right}_joint_{1..6}.pos    radians
{left,right}_gripper.pos         normalized 0..1
```

Standard LeRobot shape, so it drops into `lerobot-record` and friends:

```python
teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(...))
follower = YamUltraFollower(...)
# connect, then loop: follower.send_action(teleop.get_action())
```

It also **publishes `ik_state` back** to the relay each tick — joint angles,
engage state, and haptic intensity per arm. That's what drives controller
vibration in the headset. Haptics
mix two independent signals with `max()`: IK trouble (limit pressure, reach
error, singularity proximity) and gripper force from `send_feedback`. Either
can buzz the controller on its own.

## `single_arm_quest_teleop`

The bimanual teleop emits `left_`/`right_`-prefixed keys because that's what
the bimanual follower expects. A single-arm follower wants unprefixed
`joint_1.pos`, so this adapter wraps the bimanual one and strips the prefix
for one chosen arm.

```
--teleop.type=single_arm_quest_teleop --teleop.arm=left|right
```

Built for DAgger interventions on single-arm policies: the operator still
wears both controllers — one drives, the other stays free for the B/Y handoff
button — while the action dict landing on the robot matches the single-arm
schema.

## Operator model

- **Grip = clutch.** Hold to drive, release to reposition your hand. The arm
  holds still while released. See [`core/`](../../../core/README.md) for why
  the excess is absorbed rather than accumulated.
- **Trigger = gripper**, analog.
- **Thumbstick click** ramps the arm up from its resting pose over
  `--rest-ramp-s`. Nothing moves at startup otherwise: the teleop is seeded
  with the robot's *measured* joint positions, so the first commands hold pose.
- **The gripper self-homes once during driver init.** Keep fingers clear at
  startup — this is the one motion that happens without being asked.

## Testing without hardware

`tools/smoke_test.py` drives this package with a synthetic Quest through a
real relay and asserts the clutch, schema, and gripper behavior.
`tools/pure_sim.py` runs the real thing against a real headset with no
follower. Both are much cheaper than debugging on a powered arm — see
[`tools/`](../../../tools/README.md).
