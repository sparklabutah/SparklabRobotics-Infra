# Design notes

Why things are the way they are. The READMEs describe *what* is where; this
file holds the reasoning, the measurements behind it, and the failures that
produced it. Nothing here is needed to run the stack — it is here so the
decisions are not re-litigated or quietly undone.

## Package boundaries

`sparklab_sim` is a separate top-level package from `lerobot_robot_sparklab`
because Isaac Sim runs under its own bundled Python 3.12, which has no
`lerobot`. Importing `lerobot_robot_sparklab` under it fails at the package
`__init__`, which imports lerobot to register the robot plugins.

That makes the boundary a mechanical test rather than a matter of taste: **a
module that imports lerobot cannot live in `sparklab_sim`**, whatever it is
about. "Sim vs real" is the wrong axis. `rollout/` drives hardware and the twin
alike and belongs on the lerobot side; `yam_ultra_sim` is a lerobot plugin for
a simulated arm and belongs under `robots/`. The rollout runner and its REPL
failed that test once and were moved.

Shared assets — the URDF, meshes, rig config — stay single-source: the sim side
reads them **by path** out of the sibling package's tree (`sparklab_sim/paths.py`).
One repo, one copy of the model, two interpreters, no import coupling.


## The arms run in their own processes

i2rt's CAN thread must keep sending inside each motor's watchdog window.
In-process it shared a GIL with policy inference and checkpoint loading, got
starved, and the motors reported `loss communication` (0xD) mid-run — while
every SocketCAN fault counter read zero. The wire was clean; the frames simply
were not sent in time. Separate process, separate GIL.

`scripts/can_health.sh` is what distinguishes the two causes: nonzero error
counters mean the physical layer, all-zero counters with motors still
complaining mean host timing.

**Ownership.** The `arm_server` process owns the motors, the CAN link and the
torque, and holds no policy: the Δq clamp, gripper flip, cameras and park
timing stay follower-side. The exception is parking on its own shutdown, which
the follower cannot do if that process is killed directly.

**Transport.** One tick is two round trips per arm, issued in parallel: `read`
for joint angles and `command_clamped` for read+clamp+command under one lock.
Merging the read/command pair halved the RPC rate and closed the window in
which the arm could move between reading `present` and commanding. Known cost:
portal's client socket spins rather than blocking while a call is in flight,
about one core per client. Merging both arms into one server would halve that.

**Liveness is cached, not polled.** i2rt exposes it as `motor_chain.running`;
over RPC that would be a round trip per `is_connected` check, so every response
carries the flag. This matters because a dead chain is exactly the case you
cannot detect by asking: `get_joint_pos()` keeps returning the last pose it
read, so the robot looks healthy while every command goes nowhere.

## Teleoperation

**The frame is ground-referenced and headset-yaw aligned.** WebXR `local-floor`
is Y-up with the origin on the floor. At every clutch anchor, headset yaw is
composed with `r_calib`, so controller-forward follows the direction the
operator is facing. Pitch and roll are discarded, and turning the head while
the clutch remains held does not move the arm; the new yaw takes effect at the
next engage or re-anchor.

**The clutch re-anchors on every engage**, capturing both the controller pose
and the arm's current EE pose, so motion always resumes from where the arm
actually is. Walking around between engages changes nothing: the delta is
measured from the newest engage, never an accumulated origin.

The IK step takes the current EE pose from FK of the last *commanded* qpos, not
from the robot. Every orchestrator that changes ownership or moves the arm
directly must call `seed_qpos_from_obs()` before returning control to teleop.

**The clutch anchors on the commanded pose, never the measured one.** Measured
lags commanded by the tracking error (gravity sag, controller stiffness), and
that error differs a little every time — an engage anchored on the measurement
commands the arm back to a slightly different EE pose on every clutch press.
An earlier version seeded from the measurement on each engage and later added
a background idle-resync thread. That coupled a teleoperator to whichever
follower happened to be stored in process-global state, raced LeRobot's connect
order, and could stall the WebSocket reader during robot RPC. Resynchronization
is now an explicit orchestration step; the standalone bridge does it after
startup and every arm/disarm motion.

### Absorbing reach limits

With `rot_reach_limit` / `pos_reach_limit` active, the mapper becomes
incremental: per-tick controller deltas accumulate, and anything past the limit
is absorbed — a slipping clutch, or a mouse at the edge of the screen. Two
hardware failures this kills:

- A demand pressed far past a joint stop or gimbal can never build the ~180°
  error where the shortest-way direction flips, which showed up as cap-speed
  shaking.
- It can never wrap around and snap the wrist in from the other side. 350°
  clockwise and 10° counter-clockwise are the same orientation, and an absolute
  mapping must eventually agree with that; an incremental one never has to.

Unbounded position error had the same shape: reaching past the workspace
boundary produced cap-speed bang-bang on joints 1-3.

The trade is that absorbed motion is gone, and with `scale_rotation ≠ 1` the
per-increment gain is a *rate* gain, so curved hand paths are path-dependent on
SO(3) — a closed 60°+60° loop at 1.5× leaves ~23° of residual. Hand↔EE
correspondence therefore drifts within an engagement; re-clutching realigns.
The alternative, scaling the total delta, wraps at 360°/scale of raw twist and
reintroduces the come-around this exists to kill.

## Decoupled IK

Joints 1-3 satisfy position, joints 4-6 orientation, each one damped
least-squares step per call warm-started from the caller's qpos. The 6.2 cm
wrist non-sphericity becomes a residual EE position error when the wrist
rotates; the operator's visual feedback loop closes that gap.

`solve()` always returns a valid qpos6 — never None, never an exception. Four
boundary cases are handled in-line so the arm degrades gracefully instead of
freezing:

1. **Near the workspace boundary** the position Jacobian is ill-conditioned.
   Manipulability-adaptive damping shrinks the step smoothly to zero at the
   singularity.
2. **Near wrist gimbal lock** (θ5 → ±π/2, joint-4/6 axes aligned) the
   orientation Jacobian loses rank. The same damping recipe applies to the
   wrist sub-solve: J4/J6 steps stay bounded and the lost rotation direction is
   simply not tracked until the operator backs off. The closed-form Euler
   extraction this replaced demanded 68° of J4/J6 per 1° of target twist at
   θ5 = 89.5°.
3. **Joint limit violation** clamps elementwise into the model's limits, so the
   arm reaches as far as the joints allow.
4. **Near-antipodal orientation demand** past `rot_err_hold` parks the wrist.
   The shortest-way error direction is unstable there — it flips sign under tiny
   target jitter — and chasing it with capped steps became bang-bang oscillation
   at the Δq cap. Reported by operators as shaking when twisting far past the θ5
   stop; reproduced in sim at ~150° error with 0.2° of hand tremor.

Differential steps are continuous by construction, so the old ±2π branch-unwrap
heuristics are unnecessary.

## The vendored arm model

`yam_ultra.xml`, `yam_ultra.urdf` and the meshes are copied byte-for-byte from
i2rt v1.2.4. That release's frame-alignment pass replaced the hand-simplified
body frames with the URDF's native ones, which is why `j4_anchor` and
`LINK6_MOUNT_POS`/`QUAT` had to be recomputed rather than copied. It also fixed
an upstream bug that had joint2/joint3's upper limits swapped.

**Joint limits deviate deliberately.** `INTENTIONAL_DEVIATIONS` carries
joint3's upper limit at 3.0 rad against i2rt's nominal π. i2rt's XML is nominal
for the arm family; this rig's elbow stops around 3.0 and commanding past that
grinds it into the stop. Not drift — do not "fix" it back.

Getting limits wrong is quiet and nasty in both directions, because the IK
reads them straight off this model and both clamps into them and derives its
limit-pressure haptic from them. Too tight gives an invisible wall short of
real travel; too loose drives into a mechanical stop believing there is room.
Re-run the check after any i2rt upgrade:

    python -m lerobot_robot_sparklab.robots.yam_ultra.model.kinematics

Meshes never affect kinematics — MuJoCo builds FK and Jacobians from body
frames and joint axes — and gravity compensation runs on i2rt's model, not this
one. Keep them in sync anyway so the viewer shows the arm that is moving.

## The Isaac twin

**Kinematic, not dynamic, by scope decision.** Nothing simulates contact,
torque or grasping; the arms are posed. The policy consumes RGB plus joint
state and emits joint positions, so nothing in that loop asks the simulator
what force a gripper applies. Skipping dynamics also skips the system
identification — joint friction, damping, drive gains, contact parameters —
that none of the recorded data can supply. A policy will happily "grasp"
through the box. Right tool for checking that a checkpoint produces sane,
in-range, temporally coherent actions from real camera geometry; wrong tool for
judging whether a grasp succeeds.

**`yam_ultra_sim` is a separate robot, not `--robot.sim=true`.** That flag
already exists on the hardware follower and does something different: it spawns
kinematic `arm_server`s so the joints move with no CAN bus. It does nothing
about cameras — those stay real RealSense devices. A policy needs images, so a
rollout with no hardware needs a *rendered* rig, which is this.

**The Δq caps are shared by hardware and simulation.** The default lives in
`robots/yam_ultra/constants.py`; each config still receives its own list. An
unclamped sim silently flatters the policy — a chunk that would be cut to
0.15 rad on hardware executes in full, so the failure you are hunting cannot
reproduce.

**Parking matters between runs.** The sim server is a persistent workspace;
disconnect deliberately leaves it running so the next rollout does not pay
Isaac's ~15 s boot. That means whatever pose a run ends in is the pose the next
run starts from, and LeRobot captures it as its "initial position" and restores
to it at teardown — so one bad ending becomes permanent. Measured on a run that
ended without parking: the right arm sat 44 sigma outside the pose any demo
starts from, and the policy flailed. All joints at zero is the folded rest pose
and where every recorded episode starts; across 49 demos the mean start pose is
`left_joint_2 = 0.034`, `left_joint_3 = 0.022`, and no episode begins elsewhere.

A clean disconnect parks itself. A killed or crashed rollout never reaches that
code, which is what the server's own idle auto-park (`--auto-park-s`) covers.
Only `/step` counts as activity: polling `/obs` is watching, not driving.

**Isaac work runs on the main thread, always.** `app.update()` is main-thread
affine. Called from a request handler it does not raise — it simply never
returns, and the request hangs until the client's timeout while `/health`
carries on answering perfectly, which makes it look like a network problem
rather than a threading one. So handlers do no Isaac work: they queue a job,
block on an Event, and the main loop drains the queue, renders and wakes them.
That is also why the main loop polls at 1 ms rather than sleeping in chunks —
it is the only thread that can service a tick.

**One pump for all cameras.** Every render product advances on the same
`app.update()`, so pumping per camera does the same global work three times
over: at `settle=2` that is 6 updates a tick where 2 will do. Measured, the
difference between ~116 ms and ~40 ms per observation.

### The live viewer's fast path

The scene file is loaded as a **sublayer** of an anonymous wrapper stage rather
than opened directly:

    wrapper stage (anonymous root)
      ├─ session layer : free camera, overview camera   (never saved)
      └─ sublayer      : scene.usda                     (the file you edit)

Reloading is then `Sdf.Layer.Reload()` on that one sublayer, which recomposes
the scene while leaving everything else alone — including the Replicator render
products under `/Render`. Measured: **~0.04 s** per edit, against ~0.38 s to
tear down and reopen the stage, against a full Isaac boot (8-30 s) for the
version that called `open_stage` with live render products attached and
segfaulted every time.

Render products are prims too. Reopening or reloading the *root* layer deletes
them under Hydra's feet:

    [Error] [rtx.hydra] Invalid USD RenderProduct Prim: .../Replicator_01
    [Error] [omni.hydra] Unable to find RP Prim from previous update pass!
    Segmentation fault

Hence: cameras authored into the session layer, content edited in a sublayer,
and `Renderer.close()` before any reopen. That crash is also why
`Renderer.close()` exists at all — it used to be absent, and the viewer's
teardown probed for `close`/`destroy`/`detach` on the Renderer, found none
(`destroy` is on the underlying HydraTexture) and silently did nothing.

**`--play` does not start physics.** The timeline is *always* playing, because
`scene.start()` has to play it before Isaac will accept joint writes at all.
That used to be harmless with gravity off, but the props are now dynamic rigid
bodies, so PhysX owns their poses and writes the simulated transform back every
step. A layer reload changes the authored value and PhysX overwrites it, which
looks like "I edited the position and the viewer ignored me". `reload_layers`
therefore bounces the timeline, because stopping is what resets simulated prims
to their authored state.

### Why not Isaac's WebRTC livestream

It ships no browser client in this install (`omni.kit.livestream.webrtc` has no
HTML under it), wants a separate NVIDIA streaming app plus signal port 49100,
and carries media over UDP — which does not survive an SSH tunnel, because
`ssh -L` forwards TCP only. The MJPEG viewport is one TCP port that VS Code's
Remote-SSH forwards automatically, and the client is a browser tab.

The cost is that orbiting round-trips to the workstation. On a LAN that is
imperceptible; on a slow link it feels like dragging through treacle rather
than dropping frames.

## Rollout control

**Changing the task must also reset the engine.** The policy emits a 30-step
chunk and the loop drains it one action at a time. Swapping `engine._task`
alone leaves up to 30 already-queued actions — 1.5 s at 20 Hz, longer with
interpolation — still executing under the *old* prompt, so the arm appears to
ignore the new command and then lurch. `retarget()` clears the queue.

**Reset ramps, it does not teleport.** Home is reached by commanding an eased
trajectory over `home_duration_s`, one waypoint per tick through `send_action`,
so it still passes the follower's `max_relative_target` clamp like any policy
action. Letting the clamp rate-limit a single target instead ramps at whatever
the cap allows — 0.15 rad/tick at 30 Hz is ~4.5 rad/s — so the arm snapped home
at the fastest speed the safety bound permitted, which reads as a jolt.
Commanding the trajectory makes speed a property of *this move* rather than a
side effect of a limit; the clamp stays underneath as a backstop. Smoothstep
easing puts peak velocity at 1.5× the average, still an order of magnitude
under the clamp.

**Reset parks and holds.** Homing while the loop keeps stepping means the next
tick drives the policy on the unchanged task and the arm climbs straight back
out — which is what made `reset`/`park`/`home` look like they did nothing. All
three set `held`; a new task clears it.

**HTTP handlers never touch the robot or the policy.** They append to a queue
the control loop drains between ticks — the same discipline `policy_server`
uses, and for the same reason: neither the robot transport nor the inference
engine is thread-safe, and a blocking call from an HTTP thread deadlocks the
loop rather than failing.

## The agent harness

Two processes on purpose. The rollout holds a ~21 GB model and a live control
loop, and restarting it costs ~90 s. Editing a prompt, swapping agents or
reloading the UI must not imply restarting that. It also means a crash in the
harness cannot take the arms down — the worst case is the robot continuing on
its last instruction, which is why the loop only ever sets a task and the human
keeps pause and reset.

**What an agent may do: emit a task string and declare the goal complete.**
That is all. It cannot park, reset or stop the robot. An agent that can
retarget can already move the arms, so this is not a security boundary; it is a
blast-radius one. A confused agent picks the wrong subtask, it does not decide
to power down mid-motion.

The agent generation counter exists because a turn takes a second or two (a
model call), and a task decided *before* the operator hit stop must not be
applied *after* it — that is a reset being silently undone by a decision
already in flight.

## The relay's TLS cert

WebXR needs a secure context, so the relay must serve HTTPS and the headset
must accept the cert. Browsers have ignored the CN field for hostname
verification since Chrome 58: a cert without a `subjectAltName` is invalid for
*every* name, and the Quest browser rejects it outright rather than offering
"proceed". A bare `openssl req -x509` with default answers produces exactly
that cert, which is the trap `scripts/make_certs.sh` exists to close.

Every local IPv4 goes in the SAN, so one cert works on the lab LAN, over the
campus link, and over an `adb reverse` USB tunnel. Regenerating invalidates the
headset's acceptance, so do not re-run it casually.

H.264 is forced for video because the Quest's hardware decoder is strongest
there; VP8 is software-decoded, so higher CPU and worse latency under load.

## Measured constraints

On this rig, not assumed. Kept so they are not rediscovered.

- **A checkpoint's `config.json` is the source of truth for inference.** Do not
  re-specify dtype, chunking or amp from a script. Forcing
  `--policy.use_amp=true` onto a checkpoint that says `use_amp: false` wraps a
  bfloat16 model in fp16 autocast — that flag *is*
  `torch.autocast(device_type="cuda")`, which defaults to **float16** — giving
  narrower range (max 65504 vs 3.4e38) and a cast around every eligible op.
  Slower *and* worse actions.

- **Loop rate is arm speed.** One tick consumes one action, and a chunk is a
  trajectory the policy learned at 30 fps. A loop that misses 30 Hz plays that
  trajectory in slow motion; this is not a velocity setting. Anything costing
  per-tick time (foxglove `--display_data`, a tight `taskset`) shows up as a
  visibly slower arm. `rollout/live.py` logs `loop running slower than target`.

- **CPU pinning starved the loop it was meant to protect.** On this 24-CPU
  workstation a live rollout is ~400% CPU across 140 threads, and the two arm
  servers another ~171%, against the 600% ceiling `taskset -c 0-5` imposes.
  Measured 30 Hz unpinned against ~4 Hz pinned, same checkpoint. It was never
  intra-process isolation anyway: inference and CAN polling share one GIL,
  which affinity does not change.

- **The sync engine preprocesses every tick and discards 29 of 30.** Measured
  12.47 ms of image resize and tokenisation per pop-only tick, whose output
  `select_action` never reads. Skipping it was tried and **reverted** — it made
  the policy visibly worse, so the redundant work is load-bearing somewhere. Do
  not re-add a preprocessor bypass without measuring behaviour, not just tick
  time.

- **`max_relative_target` is a position-*step* cap, not a velocity limit.**
  Nothing bounds how fast the motors close a commanded step — that is the PD
  gains. Bounding the step is what bounds the resulting error spike, which is
  why the step is what gets capped.

- **RTC inference needs `n_action_steps / fps > inference_time`**, or it
  discards whole chunks and the arm does not move. `--inference.type=sync`
  instead stalls the loop for the whole inference — ~275 ms measured, once
  every 30 ticks.

- **Validate checkpoints before hardware.** A policy whose error exceeds the
  motion it must command will move erratically no matter how healthy the rig
  is. `scripts/analysis/validate_policy_pipeline.py` catches that in minutes
  without powering anything.
