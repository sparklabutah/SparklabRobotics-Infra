# model

The MuJoCo model the IK solves against, vendored from i2rt plus the assembly
step that makes it usable.

```
kinematics.py     builds the model, adds the IK sites, self-checks decoupling
yam_ultra.xml     the 6-joint arm, MJCF (from i2rt) — what the IK compiles
linear_4310.xml   the parallel-jaw gripper, MJCF (from i2rt)
yam_ultra.urdf    the same arm + gripper, URDF (from i2rt) — for USD/sim import
assets/*.stl      meshes, shared by both
```

The MJCF is what the IK and viewer use. The URDF is here for importers that
want URDF, and because it ships the gripper already assembled where the MJCF
needs `combine_arm_and_gripper_xml()` first. Its meshes resolve to the same
`assets/`, so nothing is duplicated.

Two naming differences when driving one from the other: URDF joints are
`dof_joint1..6` against the MJCF's `joint1..6`, and the URDF models the gripper
as two prismatic finger joints (`dof_joint7/8`, −0.04695..0) where the action
space uses one normalised `gripper.pos` 0..1.

## Provenance

| here | upstream |
|---|---|
| `yam_ultra.xml`, `yam_ultra.urdf`, `assets/{base,link1..link5}.stl` | `<i2rt>/i2rt/robot_models/arm/yam_ultra/` |
| `linear_4310.xml`, `assets/{gripper,tip_left,tip_right}.stl` | `<i2rt>/i2rt/robot_models/gripper/linear_4310/` |

Synced from **i2rt v1.2.4**, byte-for-byte identical upstream except the
entries in `INTENTIONAL_DEVIATIONS`. Take assets from the tag, not the local
clone, which sits at a pre-v1.2.4 commit whose URDF still has the swapped
joint2/joint3 limits:

```bash
git -C <i2rt> show v1.2.4:i2rt/robot_models/arm/yam_ultra/yam_ultra.urdf \
  > yam_ultra.urdf
```

## The one intentional deviation

**`joint3`'s upper limit is 3.0 rad here, against i2rt's nominal π.** Not
drift — do not "fix" it back. See
[`DESIGN.md`](../../../../../DESIGN.md#the-vendored-arm-model).

**It applies to the MJCF only.** `yam_ultra.urdf` was copied verbatim and still
carries `3.14159`, so the two assets disagree about this one joint. Harmless
while nothing loads the URDF, but a USD built from it would believe the elbow
travels ~0.14 rad past the mechanical stop. Apply the same deviation before the
URDF drives anything, or convert with the limit overridden.

## The assembly step

The YAM-Ultra ships as two MJCF fragments, and `yam_ultra.xml`'s `link6` body
is a stub because upstream i2rt sets the wrist-mount transform and grafts the
gripper subtree at runtime. `combine_arm_and_gripper_xml()` reproduces that
self-containedly: it sets link6's mount pos/quat and joint6's axis from i2rt's
`linear_4310.yml`, then appends the gripper body. Without that transform joint6
sits coaxial with joint5 and the wrist is degenerate — which reads as an IK bug
rather than a model one.

Body frames follow the URDF's native convention, so `j4_anchor` and
`LINK6_MOUNT_POS`/`QUAT` must be **recomputed** on an i2rt sync, never copied.

## The two IK sites

- **`tool0`** — the EE target, at the gripper's grasp centre. Same pose as the
  gripper XML's `grasp_site`, which i2rt's own IK uses as its TCP frame.
- **`j4_anchor`** — the position-task anchor for the decoupled IK, on link3,
  upstream of joint 4 and so wrist-invariant.

## After any i2rt upgrade, run this

```bash
python -m lerobot_robot_sparklab.robots.yam_ultra.model.kinematics
```

`_check_vendored_assets_agree()` checks the MJCF and URDF describe the same
arm; `_check_against_i2rt()` diffs against the installed i2rt and separates
expected deviations from real drift; `_self_test()` re-checks the 3+3
decoupling assumption [`../ik/`](../ik/README.md) depends on, so a model change
that quietly breaks decoupling is caught here rather than as strange arm
behaviour.

## Meshes

The `.stl` files never affect kinematics — MuJoCo builds FK and Jacobians from
body frames and joint axes — and gravity compensation runs on i2rt's model, not
this one. Keep them synced anyway so the viewer shows the arm that is moving.
