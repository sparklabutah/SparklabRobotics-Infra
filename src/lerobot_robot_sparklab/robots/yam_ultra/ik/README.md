# ik

Differential inverse kinematics for the YAM-Ultra with strict joint decoupling:
joints 1–3 satisfy position, joints 4–6 satisfy orientation. Each is one
damped-least-squares step per call, warm-started from the caller's current
`qpos`.

```
decoupled_ik.py   DecoupledIKSolver
```

Self-test: `python -m lerobot_robot_sparklab.robots.yam_ultra.ik.decoupled_ik`.

## Behaviour

`solve()` always returns a valid `qpos6` — never `None`, never an exception. A
teleop loop that can freeze mid-motion is worse than one that degrades, so four
boundary cases are handled inline: the workspace boundary, wrist gimbal lock,
joint-limit violation, and a near-antipodal orientation demand. In each case
the arm keeps moving sensibly and the operator feels *why* through haptics.

What each case does, and the hardware failures behind them, are in
[`DESIGN.md`](../../../../../DESIGN.md#decoupled-ik).

The solver publishes four 0..1 signals per tick, consumed by the teleop's
haptic mix: `last_limit_pressure`, `last_pos_err_norm`,
`last_singularity_proximity`, `last_wrist_gimbal_proximity`.

## Tunables

Damping (`lam_pos`, `lam0`, `w0`, `lam_rot`, `lam0_rot`, `w0_rot`), the
Tikhonov rest bias (`mu`, `q_rest`), the wrist park gate (`rot_err_hold`) and
the per-joint step cap (`max_dq_per_joint`) are all constructor arguments,
documented on the class. `teleop/cli.py` plumbs most of them to the CLI.

## Two things to know before changing numbers

**Limits come from the model, and are used twice.**
`DecoupledIKSolver.joint_limits` is read straight off the compiled MuJoCo model
in [`../model/`](../model/README.md), for clamping *and* for deriving the
limit-pressure haptic. Getting them wrong is quiet in both directions — read
the model README first.

**`max_dq_per_joint` is not the follower's `max_relative_target`.** Both bound
a per-tick step, but at different stages, so tuning one does not cover the
other.
