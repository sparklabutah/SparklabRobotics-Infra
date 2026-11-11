# model

The YAM-Ultra arm model, from **i2rt v1.2.4**
(`i2rt/robot_models/arm/yam_ultra/` and `.../gripper/linear_4310/`).

```
kinematics.py     builds the MuJoCo model the IK uses; run it as a self-check
yam_ultra.xml     the arm (MuJoCo)
linear_4310.xml   the gripper (MuJoCo)
yam_ultra.urdf    arm + gripper (URDF), used by the Isaac twin
assets/*.stl      meshes
```

- `tool0` is the gripper's grasp centre: the point the IK steers and `ee_pose` reports.
- Joint 3's upper limit is **3.0 rad**, not i2rt's π, because this rig's elbow stops
  there. Keep it when updating the files.

After updating from i2rt, check the files still agree:

```bash
python -m sparklab_yam.model.kinematics
```
