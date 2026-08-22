"""Build the bimanual YAM-Ultra scene: two arms, a table, a light, a camera.

Every function here must be called **after** ``SimulationApp`` exists — the
``isaacsim.*`` imports inside them do not resolve before Kit has booted. That
is why they are function-local rather than at module scope.

The arms are posed, not simulated. ``set_dof_positions`` writes joint state
directly; no physics step is required and none is taken.
"""

from __future__ import annotations

import numpy as np

from . import paths, rig

LEFT_PRIM = "/World/YamLeft"
RIGHT_PRIM = "/World/YamRight"
CAMERA_PRIM = "/World/TopCamera"
OVERVIEW_PRIM = "/World/OverviewCamera"

# 6 arm joints + 2 coupled gripper fingers, in the URDF's own order.
N_ARM_DOF = 6


def _place(stage, prim_path: str, position, yaw_rad: float) -> None:
    """Set a prim's world transform with explicit translate + Z-rotate ops.

    Clears any existing op order first: a referenced prim may have none, one,
    or an unexpected combination, and appending gives an order-dependent result.
    """
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in position)))
    xform.AddRotateZOp().Set(float(np.rad2deg(yaw_rad)))


def build(add_table: bool = True, gravity: bool = False):
    """Create the stage and return ``(left_arm, right_arm)`` Articulations.

    gravity: off by default. The URDF ships no joint drive gains, so under
        gravity the joints sag out of any pose written to them. Turn it on
        only once real drive gains exist.
    """
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.objects import DistantLight, GroundPlane
    from isaacsim.core.experimental.prims import Articulation
    from pxr import UsdPhysics

    usd = str(paths.require(paths.YAM_ULTRA_USD))

    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage()

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityMagnitudeAttr(9.81 if gravity else 0.0)

    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    DistantLight("/World/DistantLight").set_intensities(3000)

    # Same USD referenced twice — one asset, two instances. Cheaper than two
    # copies and guarantees both arms are literally the same model.
    stage_utils.add_reference_to_stage(usd, LEFT_PRIM)
    stage_utils.add_reference_to_stage(usd, RIGHT_PRIM)

    # Plain xform ops, not set_world_poses: that resolves to child prims, and
    # the fixed-base asset welds to the world once the timeline plays.
    _place(stage, LEFT_PRIM, rig.LEFT_ARM_POSITION, rig.LEFT_ARM_YAW_RAD)
    _place(stage, RIGHT_PRIM, rig.RIGHT_ARM_POSITION, rig.RIGHT_ARM_YAW_RAD)

    left = Articulation(LEFT_PRIM)
    right = Articulation(RIGHT_PRIM)

    if add_table:
        _add_table()
    return left, right


def _add_table() -> None:
    """A visual slab at z=0 standing in for the table top.

    Cosmetic only — no collision role in a kinematic scene. It makes renders
    readable and a wrong camera height obvious by eye.
    """
    from isaacsim.core.experimental.objects import Cube

    depth, width = rig.TABLE_SIZE_M
    thickness = 0.04
    Cube("/World/Table",
         positions=[[0.0, 0.0, -thickness / 2]],
         scales=[[depth, width, thickness]])


def start(app, warmup_frames: int = 5, restart: bool = False) -> None:
    """Play the timeline so joint writes are accepted, then settle the stage.

    Isaac's tensor API refuses ``set_dof_positions`` until the physics tensor
    entity exists, and that only happens once the timeline is playing ("Play
    the simulation/timeline to re-initialize it").

    ``restart`` stops the timeline first, which is what makes an edited scene
    file actually take effect. Once the timeline is playing, PhysX OWNS the
    pose of every rigid body: it writes the simulated transform back each
    step, so reloading the layer changes the authored value and nothing
    visibly moves. Editing a prop's translate and seeing it ignored is exactly
    this. Stopping resets simulated prims to their authored state, so the
    replay picks up the new file.

    This did not matter before the props became dynamic bodies -- the scene
    had gravity off and nothing was simulated, so the authored transform WAS
    the rendered one.
    """
    import omni.timeline

    tl = omni.timeline.get_timeline_interface()
    if restart:
        tl.stop()
        app.update()
    tl.play()
    for _ in range(warmup_frames):
        app.update()


# Finger travel is -0.04695..0 in the URDF; 0 is closed.
FINGER_TRAVEL = -0.04695


# PD gains per DOF, from yamlab's `controller.high_pd`. PER RADIAN, hence the
# tensor API rather than USD drives, which are per DEGREE.
DRIVE_STIFFNESS = np.array([800.0, 800.0, 800.0, 800.0, 30.0, 30.0, 2000.0, 2000.0])
DRIVE_DAMPING = np.array([50.0, 50.0, 50.0, 50.0, 5.0, 5.0, 100.0, 100.0])


def set_drive_gains(arm, stiffness=None, damping=None) -> None:
    """Give an arm's joints PD gains so drive targets actually hold.

    Without this every gain is 0, so a position target exerts no force and the
    arm does not move under ``drive()``. Call once per arm after ``start()``.
    """
    k = DRIVE_STIFFNESS if stiffness is None else np.asarray(stiffness, dtype=float)
    d = DRIVE_DAMPING if damping is None else np.asarray(damping, dtype=float)
    arm.set_dof_gains(stiffnesses=[k], dampings=[d])


def _targets(q_arm, gripper: float):
    q = np.asarray(q_arm, dtype=float).reshape(-1)
    if q.size != N_ARM_DOF:
        raise ValueError(f"expected {N_ARM_DOF} joint angles, got {q.size}")
    finger = FINGER_TRAVEL * float(np.clip(gripper, 0.0, 1.0))
    return np.concatenate([q, [finger, finger]])


def pose(arm, q_arm, gripper: float = 0.0) -> None:
    """Set one arm's joint positions directly, a kinematic write.

    Teleports the joint state, so the fingers pass through objects rather than
    pushing them. Use :func:`drive` to interact with anything.

    q_arm: 6 joint angles in radians.
    gripper: 0..1 normalised, mapped onto both prismatic finger joints.
    """
    arm.set_dof_positions([_targets(q_arm, gripper)])


def drive(arm, q_arm, gripper: float = 0.0) -> None:
    """Command one arm's joints as PD drive targets.

    Unlike :func:`pose` the joints are pulled toward the target, so contact
    builds and a gripper can hold — and the arm lags, as hardware does.
    Requires :func:`set_drive_gains`, or nothing moves.

    q_arm: 6 joint angles in radians.
    """
    arm.set_dof_position_targets([_targets(q_arm, gripper)])


# The importer nests the chain under a "Geometry" scope, so link paths are not
# <arm>/base/link1/... as the URDF's names suggest. A property of the converter.
_LINK_CHAIN = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]


def link_path(arm_prim: str, link: str) -> str:
    """Full USD path of a link in an imported arm, e.g. ``link_path(LEFT_PRIM,
    "gripper")``. Raises for unknown link names rather than returning a path
    that silently resolves to nothing."""
    if link not in _LINK_CHAIN:
        raise KeyError(f"unknown link {link!r}; known: {_LINK_CHAIN}")
    chain = _LINK_CHAIN[: _LINK_CHAIN.index(link) + 1]
    return "/".join([arm_prim, "Geometry", *chain])


def link_world_xyz(stage, arm_prim: str, link: str) -> np.ndarray:
    """World-space position of a link origin, in metres."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(link_path(arm_prim, link))
    if not prim.IsValid():
        raise LookupError(f"no prim at {link_path(arm_prim, link)}")
    xf = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.array(xf.ExtractTranslation())


def add_top_camera(resolution=(640, 360)):
    """The D435F third-person camera, centred between the arms.

    Lateral placement is measured; height and pitch are not — see ``rig``.
    Resolution defaults to the dataset's, so frames compare to recorded ones.
    """
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, UsdGeom

    stage = stage_utils.get_current_stage()
    cam = UsdGeom.Camera.Define(stage, CAMERA_PRIM)

    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(
        -rig.TOP_CAMERA_STANDOFF_M, rig.TOP_CAMERA_Y, rig.TOP_CAMERA_HEIGHT_M))
    # A USD camera looks down local -Z with +Y up; RotateX(+90) then
    # RotateZ(-90) swings it to world +X, where the arms reach.
    xform.AddRotateZOp().Set(-90.0)
    xform.AddRotateXOp().Set(float(90.0 + np.rad2deg(rig.TOP_CAMERA_PITCH_RAD)))

    cam.CreateFocalLengthAttr(rig.TOP_CAMERA_FOCAL_MM)
    return cam


def add_overview_camera(prim_path: str = OVERVIEW_PRIM, standoff: float = 2.2,
                        height: float = 1.6, pitch_deg: float = -35.0,
                        yaw_deg: float = -35.0, focal_mm: float = 15.0):
    """A wide 3/4 view of the workspace, for authoring rather than data.

    Not part of the rig. Added in memory by the live viewer after each reload,
    so it never pollutes the scene file you are editing.
    """
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, UsdGeom

    stage = stage_utils.get_current_stage()
    cam = UsdGeom.Camera.Define(stage, prim_path)
    yaw = np.deg2rad(yaw_deg)
    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(-standoff * np.cos(yaw),
                                        -standoff * np.sin(yaw), height))
    xform.AddRotateZOp().Set(-90.0 + yaw_deg)
    xform.AddRotateXOp().Set(90.0 + pitch_deg)
    cam.CreateFocalLengthAttr(focal_mm)
    return cam


def export_usd(path, add_table: bool = True) -> str:
    """Build the scene and save it as an editable ``.usda``.

    ASCII, so a prim can be moved in any text editor; ``sparklab_sim.live``
    watches the file and re-renders on save. Regenerating overwrites hand
    edits — the output is a starting point you then own.
    """
    import isaacsim.core.experimental.utils.stage as stage_utils

    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build(add_table=add_table)
    add_top_camera()
    stage_utils.save_stage(str(path))
    return str(path)


def cameras(stage, include_viewport: bool = False) -> dict:
    """Every ``UsdGeom.Camera`` in the stage, as ``{short_name: prim_path}``.

    Discovered rather than hardcoded, so a camera added by hand to the USD
    shows up in the live view.

    include_viewport: also return Kit's four stock viewport cameras, which are
        UI furniture and would otherwise cost a render product each.
    """
    from pxr import Usd, UsdGeom

    out = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.IsA(UsdGeom.Camera):
            continue
        name = prim.GetName()
        if not include_viewport and name.startswith("OmniverseKit_"):
            continue
        out[name] = prim.GetPath().pathString
    return out


# Free-look camera used by the live viewer. Authored into the session layer so
# it never reaches the .usda being edited.
FREE_PRIM = "/World/FreeCamera"

# Framed on the workspace, not the origin. Derived from rig.py so it tracks
# the measurements; the browser fetches it from /info.
HOME_VIEW = {
    "target": (0.15, 0.0, 0.25),
    "distance": 1.8,
    "azimuth_deg": -135.0,
    "elevation_deg": 22.0,
    "focal_mm": 18.0,
}


def set_orbit_camera(stage, prim_path: str = FREE_PRIM,
                     target=HOME_VIEW["target"],
                     distance: float = HOME_VIEW["distance"],
                     azimuth_deg: float = HOME_VIEW["azimuth_deg"],
                     elevation_deg: float = HOME_VIEW["elevation_deg"],
                     focal_mm: float = HOME_VIEW["focal_mm"]):
    """Point a camera at *target* from a spherical offset, creating it if absent.

    The free-look camera the browser drives. Stateless on this side: the client
    owns the orbit state and posts it on every mouse move, so there is no drift
    and a reconnecting client restores its view by replaying its own numbers.

    azimuth_deg: rotation in the XY plane from +X, world frame, +Z up.
    elevation_deg: lift off that plane, clamped shy of the poles where the
        look-at basis goes singular.
    """
    from pxr import Gf, UsdGeom

    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(np.clip(elevation_deg, -89.0, 89.0)))
    distance = max(float(distance), 1e-3)

    tgt = Gf.Vec3d(*(float(v) for v in target))
    eye = Gf.Vec3d(
        tgt[0] + distance * np.cos(el) * np.cos(az),
        tgt[1] + distance * np.cos(el) * np.sin(az),
        tgt[2] + distance * np.sin(el),
    )

    cam = UsdGeom.Camera.Define(stage, prim_path)
    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.ClearXformOpOrder()
    # SetLookAt builds a world->eye view matrix; a camera's xform is the
    # camera-to-world direction, hence the inverse.
    view = Gf.Matrix4d().SetLookAt(eye, tgt, Gf.Vec3d(0.0, 0.0, 1.0))
    xform.AddTransformOp().Set(view.GetInverse())

    cam.CreateFocalLengthAttr(float(focal_mm))
    # The default 1 cm near plane clips the table when you push in close, and
    # the default far plane is short for a room-scale overview.
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 200.0))
    return cam


def set_camera(stage, standoff=None, height=None, pitch_deg=None,
               focal_mm=None, y=None) -> dict:
    """Move the existing top camera in place. Returns the applied values.

    Mutates the prim rather than rebuilding the stage, so a warm kernel
    iterates in seconds. Omitted arguments keep their current value, so you can
    nudge one axis at a time. Write whatever converges back into ``rig.py``.
    """
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(CAMERA_PRIM)
    if not prim.IsValid():
        raise LookupError(f"no camera at {CAMERA_PRIM} — call add_top_camera() first")
    cam = UsdGeom.Camera(prim)
    xform = UsdGeom.Xformable(prim)

    # Read current values back out of USD so omitted args are true no-ops.
    cur = {op.GetOpName(): op.Get() for op in xform.GetOrderedXformOps()}
    t = cur.get("xformOp:translate", Gf.Vec3d(-rig.TOP_CAMERA_STANDOFF_M,
                                              rig.TOP_CAMERA_Y,
                                              rig.TOP_CAMERA_HEIGHT_M))
    cur_pitch = cur.get("xformOp:rotateX",
                        90.0 + np.rad2deg(rig.TOP_CAMERA_PITCH_RAD)) - 90.0

    standoff = float(-t[0] if standoff is None else standoff)
    y = float(t[1] if y is None else y)
    height = float(t[2] if height is None else height)
    pitch_deg = float(cur_pitch if pitch_deg is None else pitch_deg)

    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(-standoff, y, height))
    xform.AddRotateZOp().Set(-90.0)
    xform.AddRotateXOp().Set(90.0 + pitch_deg)

    if focal_mm is not None:
        cam.CreateFocalLengthAttr(float(focal_mm))
    focal = cam.GetFocalLengthAttr().Get()

    return {"standoff": standoff, "y": y, "height": height,
            "pitch_deg": pitch_deg, "focal_mm": float(focal)}
