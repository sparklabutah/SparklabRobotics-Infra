# core

Shared teleoperation math that is not specific to any arm.

```
pose_mapping.py   ClutchPoseMapper — controller pose → EE target
```

## The clutch model

The grip button is a clutch, like lifting a mouse off the desk:

- **Rising edge of grip** — capture the *engage frame*: the controller pose and
  the EE pose at that instant.
- **While held** — the EE target is the engaged EE pose composed with the
  controller's delta since engage, rotated from Quest world into the arm base.
- **On release** — the mapper disengages and `target()` returns `None` until
  the next engage, so the operator repositions freely and continues from
  wherever the arm actually is.

This decouples operator workspace from robot workspace: a 40 cm human reach
drives a full arm range in several strokes.

## Two mapping modes

`target()` behaves differently depending on whether you pass the arm's current
EE pose.

**Incremental + reach-limited** (production, what new code should use). Pass
the current EE pose each tick. The delta accumulates per-tick increments and is
limited to within `pos_reach_limit` / `rot_reach_limit` of where the arm is,
with the excess absorbed — a slipping clutch, so the arm never owes the
operator a debt of motion.

**Absolute since engage** (legacy). Pass no EE pose and you get the plain
delta-since-engage mapping.

The trade-offs of the absorbing mode, and the hardware failures it exists to
prevent, are in [`DESIGN.md`](../../../DESIGN.md#absorbing-reach-limits).

## Using it

Pair with [`../quest/`](../quest/README.md) for the input side:

```python
from lerobot_robot_sparklab.quest import XRFrameClient, buttons
from lerobot_robot_sparklab.core.pose_mapping import ClutchPoseMapper
```

Read grip from `buttons.GRIP`, feed the controller pose in, get an EE target
out, hand that to your robot's IK. Nothing here knows about joints, so a second
robot reuses it unchanged — `robots/yam_ultra/teleop/bi_quest_teleop.py` is a
worked example running one mapper per arm.

## Frame conventions

Input poses are WebXR: right-handed, Y up, −Z forward, quaternions
`[x, y, z, w]` (**scalar last**). MuJoCo uses `[w, x, y, z]`. The rotation into
the arm base happens here.

Self-test: `python -m lerobot_robot_sparklab.core.pose_mapping`.
