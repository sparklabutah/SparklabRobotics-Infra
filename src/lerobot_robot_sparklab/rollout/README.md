# rollout

An ordinary LeRobot rollout plus an HTTP control port, so the task can be
retargeted mid-run instead of paying the ~90 s model load again.

```
live.py   the rollout loop + the control channel
ctl.py    REPL client for it  (console script: sparklab-rollout-ctl)
```

Nothing here is sim-specific. It drives real arms and the Isaac twin equally —
the ramps go through the follower's Δq clamp exactly like a policy action,
which is why it was written for both.

## Running

```bash
./scripts/rollout.sh --mode=live          # hardware; --robot=sim for the twin
sparklab-rollout-ctl                      # second pane

rollout> pick up the scissors             # bare text retargets the policy
rollout> reset                            # ramp home, clear the action queue
rollout> park / pause / resume / status
```

The REPL is a separate process so typing is not fighting a scrolling log.
`./scripts/harness.sh` attaches a web UI and a high-level agent to the same
port — see [`../harness/README.md`](../harness/README.md).

## The control channel

```
GET  /status              task, tick count, paused, resets, uptime
GET  /observation         camera names, joint state, frame age
GET  /frame/<cam>.jpg     latest frame from that camera
POST /cmd  {"cmd":"task","arg":"pick up the scissors"}
           cmd: task | reset | park | home | pause | resume | quit
```

**Retargeting clears the policy's queued chunk.** Without that, up to 30
already-queued actions keep executing under the old instruction — the arm looks
like it ignored you, then lurches.

**`POST /cmd` only queues.** It returns before the arms have moved; commands are
applied by the loop thread between ticks. A reset ramps home over
`home_duration_s` (5 s) plus up to `reset_timeout_s` (8 s) of settle, and
`resets` in `/status` increments only once that finishes — which is the signal
to watch for completion, rather than a timer.

**`reset`, `park` and `home` leave the loop holding.** They put the arm
somewhere safe, so they stop stepping the policy and stay there; `/status`
reports `held: true`. Otherwise the very next tick steps the policy on the
unchanged task and the arm climbs straight back out of the pose you just put it
in — which made these three look like they did nothing.

A **new task releases the hold** and is the normal way to start moving again.
`resume` also releases it, as an escape hatch for continuing the current task
without retyping it.

`held` is deliberately not the same flag as `paused`: `paused` means *stopped
mid-task, the task still stands*, `held` means *at home, nothing to do until you
say so*. The operator needs to tell them apart, and a task submitted while
paused does not un-pause.

**HTTP handlers never touch the robot or the policy.** Neither the robot
transport nor the inference engine is thread-safe, so a blocking call from an
HTTP thread would deadlock the loop rather than fail. Handlers append to a
queue; the loop drains it.

Commands are drained *before* the pause check, so `reset` and `task` still work
on a paused rollout.

Frames are JPEG-encoded in the loop thread and kept latest-only, so a slow
watcher skips frames rather than building a backlog and handing an agent a
scene that no longer exists. `--publish-every N` subsamples the encode off the
control loop; `rollout.sh` passes 5.

## Reset ramps, it does not teleport

Home is reached by commanding an eased (smoothstep) trajectory over
`home_duration_s`, one waypoint per tick through `send_action` — so it still
passes the follower's `max_relative_target` clamp like any policy action. A
direct pose write would be fine in sim and violent on hardware.

**The clamp is a backstop here, not the thing setting the speed.** Sending the
final target and letting the clamp rate-limit it — which is what this used to do
— moves at whatever the cap allows: 0.15 rad/tick at 30 Hz is ~4.5 rad/s, a
3 rad move done in 0.8 s ending in a hard stop. That is a jolt. The eased ramp
peaks at ~0.9 rad/s with zero velocity at both ends, and every commanded step
stays well under the cap.

`--home-duration 0` restores the old clamp-limited behaviour; `--home-duration N`
sets any other ramp time.

On hardware `park` has no sim transport to call and falls back to the same
`go_home` ramp, so `home`, `reset` and `park` differ only in what else they do:
`reset` also resets the sim scene (a no-op on hardware) and clears the queue.
