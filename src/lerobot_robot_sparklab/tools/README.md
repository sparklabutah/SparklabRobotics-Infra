# tools

Dry-run and inspection utilities that need **no hardware powered up**. Use
these to answer "is my IK/model/teleop right?" before answering it with a
physical arm.

```
pure_sim.py     full VR teleop with no follower and no CAN
smoke_test.py   automated teleop test driven by a fake Quest
```

Both exercise the *teleop* path and say nothing about a trained policy. For
that, see `scripts/analysis/` — `validate_policy_pipeline.py` runs a checkpoint
open-loop on its own training data, `inspect_policy_chunks.py` dumps unclamped
action chunks. Also without powering anything.

## `pure_sim.py`

Develop teleoperation with the robot switched off. Runs the real teleoperator
and the real IK against the real Quest stream, but publishes `ik_state` to the
relay instead of commanding motors.

```bash
sparklab-relay                                       # terminal 1
python -m lerobot_robot_sparklab.tools.pure_sim      # terminal 2
```

The Quest UI still consumes `ik_state`. Nothing on the workstation renders it —
use the Isaac viewport (`sparklab_sim.live`) for that.

## `smoke_test.py`

The only automated test here. Stands up a fake Quest pushing synthetic
`xr_frame` messages through a real relay, with a real `BiQuestTeleoperator`
subscribed, and asserts:

- `connect()` blocks until the WebSocket is actually open
- `get_action()` returns the right dict schema, home pose first
- grip rising edge captures the engage frame, and moving the controller +X
  drives `right_joint_*`
- trigger pull drives `right_gripper.pos`
- the untouched left arm stays at home

Needs a relay running locally. It catches the class of bug — schema drift,
clutch not engaging — that is genuinely miserable to diagnose on a powered arm.
