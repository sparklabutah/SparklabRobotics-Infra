# core

Shared teleoperation math that isn't specific to any arm. Currently one
thing: clutch-relative pose mapping.

```
pose_mapping.py   ClutchPoseMapper — controller pose → EE target
```

## The clutch model

An operator's arm is short and a robot's workspace is not, so VR teleop can't
map controller pose to EE pose absolutely. Instead the **grip button is a
clutch**, exactly like lifting a mouse off the desk:

- **Rising edge of grip** — capture the *engage frame*: the controller pose
  and the EE pose at that instant.
- **While held** — the EE target is the engaged EE pose composed with the
  controller's delta since engage, rotated from Quest world coordinates into
  the arm's base frame.
- **On release** — the mapper disengages and `target()` returns `None` until
  the next engage. The operator repositions their hand freely, re-grips, and
  continues from wherever the arm actually is.

That release behavior is the whole point: it decouples operator workspace from
robot workspace, so a 40 cm human reach can drive an arm across its full
range in several strokes.

## Two mapping modes

`target()` behaves differently depending on whether you pass the arm's current
EE pose:

**Incremental + reach-limited** (production). Pass the current EE pose each
tick. The delta accumulates per-tick increments and is limited to within
`pos_reach_limit` / `rot_reach_limit` of where the arm actually is, with the
excess **absorbed** rather than accumulated — a slipping clutch. This matters
because the alternative is a target that races ahead of an arm which can't
keep up (blocked, at a joint limit, or simply slower than the operator), then
snaps when the obstruction clears. Absorbing the excess means the arm never
owes the operator a debt of motion.

**Absolute since engage** (legacy). Pass no EE pose and you get the plain
delta-since-engage mapping. Simpler, but exhibits exactly the racing-ahead
behavior above.

New code should pass the current EE pose.

## Using it

Pair with [`../quest/`](../quest/README.md) for the input side:

```python
from lerobot_robot_sparklab.quest import XRFrameClient, buttons
from lerobot_robot_sparklab.core.pose_mapping import ClutchPoseMapper
```

Read grip from `buttons.GRIP`, feed the controller pose in, get an EE target
out, then hand that to your robot's own IK. Nothing here knows about joints,
so a second robot reuses it unchanged — see
`robots/yam_ultra/teleop/bi_quest_teleop.py` for a worked example (it runs one
mapper per arm).

## Frame conventions

Input poses are WebXR: right-handed, Y up, −Z forward, quaternions
`[x, y, z, w]` (**scalar last**). MuJoCo uses `[w, x, y, z]`. The rotation
into the arm's base frame happens here; getting the quaternion order wrong is
the classic bug, and it looks like a plausible-but-wrong rotation rather than
an obvious failure.
