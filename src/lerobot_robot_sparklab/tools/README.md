# tools

Standalone entry points that are not part of the LeRobot CLI surface.

```
pure_sim.py       full VR teleop with no follower and no CAN
smoke_test.py     automated teleop test driven by a fake Quest
record_manual.py  dataset recording with operator-keyed episode boundaries
```

`pure_sim.py` and `smoke_test.py` need no hardware powered up; both exercise
the *teleop* path and say nothing about a trained policy. For that see
`scripts/analysis/` — also without powering anything.

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

## `record_manual.py`

`lerobot-record` with the episode/reset timers replaced by keyboard control.
Same `--robot.* / --teleop.* / --dataset.*` flags; `episode_time_s` and
`reset_time_s` are ignored. Launched via `scripts/record_manual.sh`.

| key | action |
|---|---|
| space, or B/Y on either controller | start episode / end episode (saved, then arms park to zeros) |
| r | discard the episode in progress (also parks) |
| q / Esc | quit — a half-recorded episode is discarded |

Teleop stays live between episodes; frames are only written while an episode
is open. Every episode end ramps the arms to the all-zeros park pose, so each
episode starts from the same configuration.

Control runs at `--control_hz` (default 120, must be an integer multiple of
`dataset.fps`) with dataset frames taken every Nth tick — the teleop's filters
and per-tick Δq caps assume a fast loop. The recorded action is the follower's
Δq-clamped *sent* action, not the raw teleop command.

`--log_xr` (default true) adds XR debug features per frame: `observation.xr.
{left,right,headset}_pose` (Quest local-floor position + xyzw quaternion + a
valid flag), `..{left,right}_buttons` (analog values, padded to 16), and
`..state` (engaged flags + frame age). Poses are raw controller readouts —
enough to re-run the pose mapping offline. A resumed dataset without these
columns records without them. A stats block (episodes, frames, mean length, loop Hz)
reprints at every episode start. `--dataset.num_episodes` acts as a session
cap; `--resume=true` appends to an existing dataset.
