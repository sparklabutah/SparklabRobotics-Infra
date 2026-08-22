# tools

Dry-run and inspection utilities that need no hardware powered up.

```
pure_sim.py     full VR teleop with no follower and no CAN
smoke_test.py   automated teleop test driven by a fake Quest
```

Both exercise the *teleop* path and say nothing about a trained policy. For
that see `scripts/analysis/` — also without powering anything.

## `pure_sim.py`

Runs the real teleoperator and the real IK against the real Quest stream, but
publishes `ik_state` to the relay instead of commanding motors.

```bash
sparklab-relay                                       # terminal 1
python -m lerobot_robot_sparklab.tools.pure_sim      # terminal 2
```

The Quest UI consumes `ik_state` directly. To watch it on the workstation, use
the Isaac viewport (`sparklab_sim.live`).

## `smoke_test.py`

Stands up a fake Quest pushing synthetic `xr_frame` messages through a real
relay, with a real `BiQuestTeleoperator` subscribed, and asserts:

- `connect()` blocks until the WebSocket is open
- `get_action()` returns the right dict schema, home pose first
- a grip rising edge captures the engage frame, and moving the controller +X
  drives `right_joint_*`
- trigger pull drives `right_gripper.pos`
- the untouched left arm stays at home

Needs a relay running locally.
