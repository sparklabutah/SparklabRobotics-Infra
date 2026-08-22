# ik

Differential inverse kinematics for the YAM-Ultra, with **strict joint
decoupling**: joints 1–3 satisfy position, joints 4–6 satisfy orientation.
Each is one damped-least-squares step per call, warm-started from the caller's
current `qpos`.

```
decoupled_ik.py   DecoupledIKSolver
```

## Why decoupled rather than a general solver

i2rt ships its own IK now, and we don't use it. The reason is the loop this
sits in: VR teleoperation at interactive rates with a human in the feedback
path. A decoupled 3+3 solve is two small well-conditioned problems instead of
one 6×6, it warm-starts from the current pose so each tick is a *step* rather
than a solve-from-scratch, and it has no iteration-count variance — the cost
per tick is flat and predictable.

The tradeoff is real and deliberate: **the YAM-Ultra's wrist is not
spherical** (6.2 cm offset), so decoupling leaves a residual EE position error
when the wrist rotates. We can afford that because the operator is watching
the arm and closing the gap continuously — the same reason you can drive a car
with imprecise steering. A policy replaying recorded joint targets never
invokes this code at all, so the residual never reaches the trained behavior.

If you add a robot with a genuinely spherical wrist, this decoupling is exact
and strictly cheaper. If you add one where the operator is *not* in the loop,
prefer a full solver.

## Never returns `None`

The solver always produces a valid `qpos6`. A teleop loop that can freeze
mid-motion is worse than one that degrades, so four boundary cases are handled
inline:

1. **Workspace boundary** — the position Jacobian goes ill-conditioned.
   Manipulability-adaptive damping shrinks step magnitude smoothly to zero at
   the singularity instead of blowing up.
2. **Wrist gimbal lock** (θ5 → ±π/2, joint-4 and joint-6 axes align) — the
   orientation Jacobian loses rank. Same adaptive damping on the wrist
   sub-solve; the lost rotation direction is simply not tracked until the
   operator backs off.
3. **Joint limit violation** — elementwise clamp into the model's limits. The
   arm reaches as far as the joints allow and the residual EE error is left to
   the operator.
4. **Near-antipodal orientation demand** — past `rot_err_hold` the
   shortest-way error direction is unstable, so the wrist parks and reports
   saturating limit pressure. Tracking resumes once the operator backs off
   below the gate.

In every case the arm keeps moving sensibly and the operator feels *why*
through haptics, rather than the arm stopping with no explanation.

## Limits come from the model, and matter twice

`DecoupledIKSolver.joint_limits` is read straight off the compiled MuJoCo
model in [`../model/`](../model/README.md), and it is used for two different
things: clamping, and deriving the limit-pressure haptic. Getting them wrong
is quiet in both directions — too tight and the arm hits an invisible wall
short of its travel; too loose and the IK drives confidently into a mechanical
stop. See the model README before changing any number here.

## `max_dq_per_joint`

Applied last, as a per-joint bound on the step. This is the operator-safety
layer against residual fast motion, and it is separate from the follower's own
`max_relative_target` clamp — the two guard different stages, so don't assume
tuning one covers the other.
