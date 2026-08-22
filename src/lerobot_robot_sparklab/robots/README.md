# robots

One subpackage per robot. Each owns everything specific to it: kinematics and
MJCF, IK, the bridge to its motor driver, its LeRobot `Robot` adapter, its
teleoperator, and its rig config (camera serials, CAN channels).

```
yam_ultra/    bimanual i2rt YAM-Ultra — the first one
```

Anything reusable belongs one level up in `relay/`, `quest/`, `core/`,
`cameras/` or `tools/`. If shared code needs a robot-specific tweak, push the
robot-specific part down here rather than branching upstairs.

## Adding a robot

1. `mkdir <name>/`
2. Write a `Robot` subclass and a config registered with
   `@RobotConfig.register_subclass("<name>")`.
3. Re-export both from `<name>/__init__.py`.
4. Add `from .robots import <name>` to the package `__init__.py`.

Then `--robot.type=<name>` works in every `lerobot-*` CLI, with no wrapper
script.

For teleoperation, reuse `quest/` (controller poses and buttons) and `core/`
(clutch-relative mapping) and write only the part mapping an EE target onto
your robot's action space.

Steps 3 and 4, and the distribution's `lerobot_robot_` prefix, are load-bearing
for plugin discovery. Missing the re-export produces

```
ImportError: Could not locate device class '<Name>' for config '<Name>Config'
```

which points at the class rather than the missing re-export. See
[`DESIGN.md`](../../../DESIGN.md#two-naming-rules-that-are-load-bearing).

## What tends to be robot-specific

Kinematic model and IK, joint and velocity limits, action and observation
feature schemas, the motor driver and its process model, camera identity, and
home/park poses. Transport, clutch semantics, button layout and camera tuning
are already shared.
