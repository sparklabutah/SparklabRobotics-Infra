# robots

One subpackage per robot in the lab. Each owns everything specific to it:
kinematics/MJCF, IK, the bridge to its motor driver, its LeRobot `Robot`
adapter, its teleoperator, and its rig config (camera serials, CAN channels).

```
yam_ultra/    bimanual i2rt YAM-Ultra — the first one
```

Anything genuinely reusable belongs one level up in `relay/`, `quest/`,
`core/`, `cameras/` or `tools/` instead. **If shared code needs a
robot-specific tweak, that's the signal something leaked** — push the
robot-specific part down here rather than adding a branch upstairs.

## Adding a robot

1. `mkdir <name>/`
2. Write a `Robot` subclass and a config registered with
   `@RobotConfig.register_subclass("<name>")`.
3. **Re-export both from `<name>/__init__.py`.** Not optional — see below.
4. Add `from .robots import <name>` to the package `__init__.py` so importing
   the distribution runs your registration decorator.

Then `--robot.type=<name>` works in every `lerobot-*` CLI, with no wrapper
script.

For teleoperation, reuse `quest/` (controller poses and buttons) and `core/`
(clutch-relative mapping) and write only the part that maps an EE target onto
your robot's action space. That mapping is genuinely robot-specific; the rest
is not.

## The two rules that will cost you an hour

**The distribution name must keep its `lerobot_robot_` prefix.** Every
`lerobot-*` CLI calls `register_third_party_plugins()`, which scans installed
*distribution* names for that prefix and imports the matching top-level
module. That import is what runs your decorators. Rename the distribution and
every robot in the lab silently stops being a valid `--robot.type`.

**Each robot must re-export its Robot and Config from its own
`__init__.py`.** Once a config is selected, LeRobot's
`make_device_from_device_class` looks for the Robot class **only in the direct
parent package of the config's module**. It does not walk up to the package
root, so a re-export at the top level does not satisfy it. Miss this and you
get:

```
ImportError: Could not locate device class '<Name>' for config '<Name>Config'
```

which points at the class rather than at the missing re-export, and reads like
a much deeper problem than it is.

## What tends to be robot-specific

Worth deciding deliberately rather than discovering later: kinematic model and
IK, joint/velocity limits, action and observation feature schemas, the motor
driver and its process model, camera identity, and home/park poses. Transport,
clutch semantics, button layout, and camera *tuning* are not — those are
already shared.
