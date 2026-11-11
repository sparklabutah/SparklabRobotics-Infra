# ik

Inverse kinematics for the YAM-Ultra: a target gripper pose (`tool0` in the arm
base) to joint angles.

```
solver.py   IKSolver
```

Each `solve()` call takes one damped least-squares step from the joint angles it
is given, and always returns 6 joint angles within the limits. Teleop calls it
every tick; `coding_agent_sparklab.robot.ik` iterates it to convergence. Its
tuning (damping, `rot_weight`, the per-step joint cap) is in the constructor's
docstring.

The decoupled solver it replaced is in [`archive/`](../../../../archive/README.md).
