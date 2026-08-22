# rollout

An ordinary LeRobot rollout plus an HTTP control port, so the task can be
retargeted mid-run instead of paying the ~90 s model load again.

```
live.py   the rollout loop + the control channel
ctl.py    REPL client for it  (console script: sparklab-rollout-ctl)
```

Nothing here is sim-specific: it drives real arms and the Isaac twin equally,
and the ramps go through the follower's Δq clamp exactly like a policy action.

## Running

```bash
./scripts/rollout.sh --mode=live          # hardware; --robot=sim for the twin
sparklab-rollout-ctl                      # second pane

rollout> pick up the scissors             # bare text retargets the policy
rollout> reset                            # ramp home, clear the action queue
rollout> park / pause / resume / status
```

`./scripts/harness.sh` attaches a web UI and a high-level agent to the same
port — see [`../harness/README.md`](../harness/README.md).

## The control channel

```
GET  /status              task, tick count, paused, held, resets, uptime
GET  /observation         camera names, joint state, frame age
GET  /frame/<cam>.jpg     latest frame from that camera
POST /cmd  {"cmd":"task","arg":"pick up the scissors"}
           cmd: task | reset | park | home | pause | resume | quit
```

**Retargeting clears the policy's queued chunk**, so a new task lands on the
next tick rather than after up to 30 stale actions.

**`POST /cmd` only queues.** It returns before the arms have moved; commands
are applied by the loop thread between ticks. A reset ramps home over
`home_duration_s` (5 s) plus up to `reset_timeout_s` (8 s) of settle, and
`resets` in `/status` increments only once that finishes — watch that for
completion rather than a timer.

**`reset`, `park` and `home` leave the loop holding.** They put the arm
somewhere safe and stay there; `/status` reports `held: true`. A new task
releases the hold and is the normal way to start moving again; `resume` also
releases it, for continuing the current task without retyping it.

`held` is not the same flag as `paused`: `paused` means *stopped mid-task, the
task still stands*, `held` means *at home, nothing to do until you say so*. A
task submitted while paused does not un-pause.

**HTTP handlers never touch the robot or the policy** — they append to a queue
the loop drains between ticks. Commands are drained *before* the pause check,
so `reset` and `task` still work on a paused rollout.

Frames are JPEG-encoded in the loop thread and kept latest-only.
`--publish-every N` subsamples the encode off the control loop; `rollout.sh`
passes 5.

## Homing

Home is reached by commanding an eased (smoothstep) trajectory over
`home_duration_s`, one waypoint per tick through `send_action`, peaking at
~0.9 rad/s with zero velocity at both ends. `--home-duration 0` restores the
older behaviour of sending the final target and letting the Δq clamp rate-limit
it; `--home-duration N` sets any other ramp time. The reasoning is in
[`DESIGN.md`](../../../DESIGN.md#rollout-control).

On hardware `park` has no sim transport to call and falls back to the same
ramp, so `home`, `reset` and `park` differ only in what else they do: `reset`
also resets the sim scene (a no-op on hardware) and clears the queue.
