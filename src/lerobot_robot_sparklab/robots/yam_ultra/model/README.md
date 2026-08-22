# model

The MuJoCo model the IK actually solves against, vendored from i2rt plus the
assembly step that makes it usable.

```
kinematics.py     builds the model, adds the IK sites, self-checks decoupling
yam_ultra.xml     the 6-joint arm, MJCF (from i2rt) — what the IK compiles
linear_4310.xml   the parallel-jaw gripper, MJCF (from i2rt)
yam_ultra.urdf    the same arm + gripper, URDF (from i2rt) — for USD/sim import
assets/*.stl      meshes, shared by both
```

The MJCF is the one the IK and the viewer use. The URDF is here for importers
that want URDF (Isaac Sim's is the better-trodden path than its MJCF one) and
because it ships the gripper **already assembled**, where the MJCF needs
`combine_arm_and_gripper_xml()` first. Its meshes resolve to the same
`assets/` directory, so nothing is duplicated.

Two naming differences to map explicitly when driving one from the other:
URDF joints are `dof_joint1..6` against the MJCF's `joint1..6`, and the URDF
models the gripper as **two prismatic finger joints** (`dof_joint7/8`,
−0.04695..0) where the action space uses one normalized `gripper.pos` 0..1.

## Provenance

Copied from i2rt's own models:

| here | upstream |
|---|---|
| `yam_ultra.xml`, `yam_ultra.urdf`, `assets/{base,link1..link5}.stl` | `<i2rt>/i2rt/robot_models/arm/yam_ultra/` |
| `linear_4310.xml`, `assets/{gripper,tip_left,tip_right}.stl` | `<i2rt>/i2rt/robot_models/gripper/linear_4310/` |

Synced from **i2rt v1.2.4**, byte-for-byte identical to upstream *except* the
entries in `INTENTIONAL_DEVIATIONS`. Take assets from the tag, not from the
local clone — it sits at a pre-v1.2.4 commit whose URDF still has the swapped
joint2/joint3 limits:

```bash
git -C <i2rt> show v1.2.4:i2rt/robot_models/arm/yam_ultra/yam_ultra.urdf \
  > yam_ultra.urdf
```

Vendoring is deliberate: a driver upgrade cannot move the sim and the IK at the
same time as it changes motor behaviour, which makes an i2rt version bump a
single-variable test instead of a guess.

## The one intentional deviation

**`joint3` upper limit is 3.0 rad here, against i2rt's nominal 3.14159 (π).**
Not drift — do not "fix" it back. i2rt's XML is nominal for the whole arm
family; the elbow on *this physical rig* stops around 3.0, and commanding past
that grinds it into the stop.

**It applies to the MJCF only.** `yam_ultra.urdf` was copied verbatim and still
carries the nominal `3.14159`, so the two assets disagree about this one joint.
Harmless while nothing loads the URDF, but a USD built from it would believe
the elbow travels ~0.14 rad past the mechanical stop. Apply the same deviation
before the URDF drives anything, or convert with the limit overridden.

Worth re-measuring: the ~0.14 rad gap is measured against a target that was
only corrected in v1.2.4, so it may have shrunk or closed. Measure the physical
stop before assuming it is still real.

## Why the assembly step exists

The YAM-Ultra ships as two MJCF fragments. `yam_ultra.xml`'s `link6` body is a
**stub** — identity pos/quat, placeholder inertial — because upstream i2rt
sets the real wrist-mount transform and grafts the gripper subtree at runtime.

`combine_arm_and_gripper_xml()` reproduces that self-containedly: it sets
link6's mount pos/quat and joint6's axis from i2rt's `linear_4310.yml`
(`last_joint_mount.yam_ultra`), then appends the gripper body. **Without that
transform joint6 sits exactly coaxial with joint5** and the wrist is
degenerate — which looks like an IK bug, not a model bug.

Body frames follow the URDF's native convention, not hand-simplified
identity/axis-aligned ones, so `j4_anchor` and
`LINK6_MOUNT_POS`/`LINK6_MOUNT_QUAT` must be **recomputed** on an i2rt sync,
never copied.

## The two IK sites

- **`tool0`** — the EE target, at the gripper's grasp center. Same pose as the
  gripper XML's `grasp_site`, which i2rt's own IK uses as its TCP frame.
- **`j4_anchor`** — the position-task anchor for the decoupled IK, on link3.

`_self_test` numerically re-checks the 3+3 decoupling assumption that
[`../ik/`](../ik/README.md) depends on, so a model change that quietly breaks
decoupling gets caught here rather than as strange arm behavior.

## After any i2rt upgrade, run this

```bash
python -m lerobot_robot_sparklab.robots.yam_ultra.model.kinematics
```

`_check_against_i2rt()` diffs this model against the installed i2rt and
separates expected deviations from real drift. Limits are read straight into
`DecoupledIKSolver.joint_limits` and used for both clamping *and* the
limit-pressure haptic, so silent drift here is expensive in both directions.

## Meshes

The `.stl` files never affect kinematics — MuJoCo builds FK and Jacobians from
body frames and joint axes — and gravity compensation runs on i2rt's model,
not this one. Keep them synced anyway, so the viewer shows the arm that is
actually moving.
