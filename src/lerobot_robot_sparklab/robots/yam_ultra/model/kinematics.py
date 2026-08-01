"""Mujoco model construction for the I2RT YAM-Ultra arm.

Rebuilds the two named sites the IK pipeline needs (see below) from the
YAM-Ultra's own MJCF, and numerically re-checks the 3+3 joint-decoupling
assumption `ik/decoupled_ik.py` relies on (see ``_self_test``).

The YAM-Ultra ships as two MJCF fragments, both vendored alongside this
file:

  yam_ultra.xml     — the 6-joint arm. Its ``link6`` body is a stub
                      (identity pos/quat, placeholder inertial): upstream
                      i2rt sets the real wrist-mount transform and grafts
                      the gripper subtree at runtime.
  linear_4310.xml   — the stock parallel-jaw gripper (meshes shared with
                      linear_3507; only the inertial mass differs).

PROVENANCE — copied from i2rt's own models:

  yam_ultra.xml, assets/{base,link1..link5}.stl
      <i2rt>/i2rt/robot_models/arm/yam_ultra/
  linear_4310.xml, assets/{gripper,tip_left,tip_right}.stl
      <i2rt>/i2rt/robot_models/gripper/linear_4310/

Everything matches i2rt byte-for-byte EXCEPT the deliberate deviations in
``INTENTIONAL_DEVIATIONS`` below — currently joint3's upper limit, which is
3.0 rad here against i2rt's nominal 3.14159 (π). **This is not drift; do not
"fix" it back.** i2rt's XML is the nominal model for the arm family; the
elbow on this physical rig stops around 3.0 and commanding past that just
grinds it into the stop.

Synced from i2rt v1.2.4 (PR #46, "YAM family URDF/MJCF frame alignment").
Before that release, i2rt's own yam_ultra.xml had joint2 and joint3's upper
limits swapped (joint2 0..π, joint3 0..3.66519) — a bug in their hand-authored
MJCF, not in this rig. The old INTENTIONAL_DEVIATIONS entry compared our 3.0
against their mislabeled 3.66519; the real nominal for joint3 was always π,
much closer to what this rig measures. Re-measure the physical stop before
assuming the ~0.14 rad (~8°) gap to π is still real — it may have shrunk or
closed now that the comparison target is correct. The v1.2.4 alignment pass
also changed the arm's internal body-frame convention (joint axes and body
quats are no longer hand-simplified to identity/axis-aligned; they now carry
the URDF's native frames), which is why the ``j4_anchor`` site offset and
``LINK6_MOUNT_POS``/``LINK6_MOUNT_QUAT`` below needed recomputing, not just
copying, when this sync happened — see their comments.

Getting these limits wrong is quiet and nasty in both directions, because
the IK reads them straight off this model (``DecoupledIKSolver.joint_limits``)
and both clamps into them and derives the limit-pressure haptic from them:
too tight and the arm hits an invisible wall short of its travel; too loose
and the IK happily drives it into a mechanical stop believing it has room.
``_check_against_i2rt()`` (run it via ``python -m lerobot_robot_sparklab.robots.yam_ultra.model.kinematics``) diffs against
the installed i2rt and separates expected deviations from real drift, so
re-run it after any i2rt upgrade.

The meshes never affect kinematics (MuJoCo builds FK/Jacobians from body
frames and joint axes) and gravity compensation runs on i2rt's own model,
not this one — but keep them in sync anyway so the viewer shows the arm
that is actually moving.

``combine_arm_and_gripper_xml()`` reproduces the upstream i2rt combine
step self-containedly: it sets link6's mount pos/quat and joint6's axis
from i2rt's ``linear_4310.yml`` (``last_joint_mount.yam_ultra`` entry),
then appends the gripper body to link6. Without that mount transform
joint6 would sit exactly coaxial with joint5 (the stub's identity quat)
and the wrist would be degenerate.

The compiled model is then decorated with the two named sites the IK
pipeline needs:

  tool0      — the end-effector target, placed at the gripper's grasp
               center (same pose as the gripper XML's ``grasp_site``,
               which upstream i2rt's own IK uses as the TCP frame).
  j4_anchor  — the position-task anchor for the decoupled IK. On link3
               (upstream of joint 4 → wrist-invariant), offset 10 cm past
               joint 4's pivot along the link3→link4 direction, keeping
               the anchor off the joint-1 axis at folded poses.

Decoupling assumption (checked numerically, see ``_self_test``): joints
1-3 are yaw + two parallel pitches (position), joints 4-6 are pitch ⊥
yaw ⊥ roll (orientation). Joints 5 and 6 intersect; joint 4 sits 7.0 cm
off the wrist center — a modest non-sphericity, absorbed by the
operator's visual feedback loop.

The IK math itself lives in ik/decoupled_ik.py.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).resolve().parent

ARM_XML = _HERE / "yam_ultra.xml"
GRIPPER_XML = _HERE / "linear_4310.xml"

# Wrist-mount transform for the linear gripper on the YAM-Ultra,
# transcribed from i2rt's robots/config/linear_4310.yml
# (last_joint_mount.yam_ultra). Applied to the link6 stub: pos/quat
# replace the identity placeholders, and joint6's axis becomes the tool
# roll axis (the fingers' approach direction, -y in link5's frame).
#
# As of i2rt v1.2.4's URDF/MJCF alignment pass (see INTENTIONAL_DEVIATIONS
# below), the arm-only yam_ultra.xml now bakes this same transform directly
# into its link6 body — link6 is no longer an identity stub. Overwriting it
# here with these (matching) values is now a no-op rather than the only
# source of truth, but kept for parity with i2rt's own combine step and so
# this still works against an arm XML that predates the alignment pass.
LINK6_MOUNT_POS = "0.0404996 2.39858e-07 -0.0419481"
LINK6_MOUNT_QUAT = "1 0 0 0"
JOINT6_AXIS = "0 0 -1"

# Grasp-center offset in the gripper body frame, matching the gripper
# XML's grasp_site (the TCP frame upstream i2rt's IK solves for). The
# quat flips z to point out along the fingers.
TOOL0_OFFSET_XYZ = np.array([0.0, 0.0, -0.1347])
TOOL0_OFFSET_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

DEFAULT_Q_REST = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])

# Places where this rig's model deliberately departs from i2rt's nominal
# YAM-Ultra XML: joint name -> ((lower, upper), why). _check_against_i2rt()
# reports these as expected instead of drift, so the check stays meaningful.
# Measured on the physical arm — i2rt's XML describes the arm family, not
# the travel this particular rig actually has.
INTENTIONAL_DEVIATIONS: dict[str, tuple[tuple[float, float], str]] = {
    "joint3": ((0.0, 3.0),
               "elbow stops near 3.0 rad on this rig; i2rt's nominal 3.14159 "
               "(pi, corrected in v1.2.4 — see module docstring) would let "
               "the IK drive it into the mechanical stop"),
}


def combine_arm_and_gripper_xml(
    arm_xml: str | Path = ARM_XML,
    gripper_xml: str | Path = GRIPPER_XML,
) -> str:
    """Combine the arm and gripper MJCF fragments into one XML string.

    Self-contained port of i2rt's ``combine_arm_and_gripper_xml``: set
    link6's wrist-mount pos/quat and joint6's axis, merge the gripper's
    mesh assets (absolutized paths), append the ``<body name="gripper">``
    subtree to link6, and carry over the gripper's equality/contact
    sections (the coupled finger slide joints).
    """
    arm_xml, gripper_xml = Path(arm_xml), Path(gripper_xml)
    arm_root = ET.parse(arm_xml).getroot()
    grip_root = ET.parse(gripper_xml).getroot()

    link6 = arm_root.find(".//body[@name='link6']")
    if link6 is None:
        raise RuntimeError("link6 body not found in arm XML")
    link6.set("pos", LINK6_MOUNT_POS)
    link6.set("quat", LINK6_MOUNT_QUAT)
    joint6 = link6.find("joint")
    if joint6 is None:
        raise RuntimeError("joint6 not found on link6")
    joint6.set("axis", JOINT6_AXIS)

    # Absolutize mesh paths (arm and gripper may live apart from the cwd
    # the string is later compiled from), then drop the meshdir.
    for root, xml_path in ((arm_root, arm_xml), (grip_root, gripper_xml)):
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        asset = root.find("asset")
        if asset is not None:
            for child in asset:
                file = child.get("file")
                if file and not os.path.isabs(file):
                    child.set("file", str((xml_path.parent / meshdir / file).resolve()))
        if compiler is not None and compiler.get("meshdir"):
            del compiler.attrib["meshdir"]

    # Merge gripper assets into the arm's asset block (skip duplicates).
    arm_asset = arm_root.find("asset")
    grip_asset = grip_root.find("asset")
    if grip_asset is not None:
        if arm_asset is None:
            arm_asset = ET.Element("asset")
            arm_root.insert(0, arm_asset)
        existing = {(c.tag, c.get("name")) for c in arm_asset}
        for child in grip_asset:
            if (child.tag, child.get("name")) not in existing:
                arm_asset.append(deepcopy(child))

    grip_body = grip_root.find(".//body[@name='gripper']")
    if grip_body is None:
        raise RuntimeError("gripper body not found in gripper XML")
    link6.append(deepcopy(grip_body))

    for section_tag in ("equality", "contact"):
        grip_section = grip_root.find(section_tag)
        if grip_section is None:
            continue
        arm_section = arm_root.find(section_tag)
        if arm_section is None:
            arm_section = ET.SubElement(arm_root, section_tag)
        for child in grip_section:
            arm_section.append(deepcopy(child))

    return ET.tostring(arm_root, encoding="unicode")


def build_spec_with_tool0_site(
    arm_xml: str | Path | None = None,
    gripper_xml: str | Path | None = None,
) -> mujoco.MjSpec:
    """The combined arm+gripper spec, decorated with the two IK sites, but
    not yet compiled.

    Split out from ``build_model_with_tool0_site`` so tooling can add to the
    model before compiling (``inspect_model.py`` bolts on position actuators
    to get one slider per joint in the MuJoCo UI) without that ever touching
    the model the IK solves against — the solver keeps compiling the spec
    exactly as returned here, actuator-free.
    """
    xml = combine_arm_and_gripper_xml(
        arm_xml or ARM_XML, gripper_xml or GRIPPER_XML
    )
    spec = mujoco.MjSpec.from_string(xml)

    gripper = spec.body("gripper")
    if gripper is None:
        raise RuntimeError("gripper body missing after combine")
    gripper.add_site(
        name="tool0",
        pos=TOOL0_OFFSET_XYZ.tolist(),
        quat=TOOL0_OFFSET_QUAT.tolist(),
    )
    # Position-task anchor. On link3 (upstream of joint 4 → fully
    # wrist-invariant), 10 cm past joint 4's pivot along the link3→link4
    # direction. Joint 4 origin in link3 frame (link4 body pos):
    # (-0.0600, 0.0678, -0.2450), magnitude 0.2612 m. Unit direction
    # (-0.2297, 0.2596, -0.9380); 10 cm offset → anchor at
    # (-0.0830, 0.0938, -0.3388). (Recomputed for i2rt v1.2.4's
    # URDF/MJCF-aligned yam_ultra.xml — link3's local frame convention
    # changed along with the joint-limit fix in INTENTIONAL_DEVIATIONS
    # below, so this is not the same number as before even though the
    # physical anchor point it targets hasn't moved.)
    link3 = spec.body("link3")
    if link3 is None:
        raise RuntimeError("link3 body not found in arm XML")
    link3.add_site(
        name="j4_anchor",
        pos=[-0.0830, 0.0938, -0.3388],
        size=[0.015, 0.0, 0.0],     # 1.5 cm sphere
        rgba=[1.0, 0.5, 0.0, 1.0],  # orange
    )
    return spec


def build_model_with_tool0_site(
    arm_xml: str | Path | None = None,
    gripper_xml: str | Path | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = build_spec_with_tool0_site(arm_xml, gripper_xml)
    model = spec.compile()
    return model, mujoco.MjData(model)


# ---------- self-test ----------

def _joint_ranges(xml_path: Path) -> dict[str, tuple[float, float]]:
    """joint name -> (lower, upper) as written in an MJCF file."""
    out: dict[str, tuple[float, float]] = {}
    for j in ET.parse(xml_path).getroot().iter("joint"):
        name, rng = j.get("name"), j.get("range")
        if name and rng:
            lo, hi = (float(v) for v in rng.split())
            out[name] = (lo, hi)
    return out


def _check_against_i2rt() -> None:
    """Diff the vendored MJCF against i2rt's installed copy, if importable.

    Separates the deliberate deviations in ``INTENTIONAL_DEVIATIONS`` from
    genuine drift, so the expected joint3 difference doesn't train everyone
    to ignore this check. Skipped when i2rt isn't installed (the model works
    standalone); never raises — it reports, you decide.
    """
    try:
        import i2rt
    except ImportError:
        print("i2rt not installed — skipping vendored-model drift check")
        return

    root = Path(i2rt.__file__).resolve().parent / "robot_models"
    their_arm = root / "arm" / "yam_ultra" / "yam_ultra.xml"

    # Meshes + the gripper XML carry no intentional deviations: byte-compare.
    pairs = [(GRIPPER_XML, root / "gripper" / "linear_4310" / "linear_4310.xml")]
    for name in ("base", "link1", "link2", "link3", "link4", "link5"):
        pairs.append((_HERE / "assets" / f"{name}.stl",
                      root / "arm" / "yam_ultra" / "assets" / f"{name}.stl"))
    for name in ("gripper", "tip_left", "tip_right"):
        pairs.append((_HERE / "assets" / f"{name}.stl",
                      root / "gripper" / "linear_4310" / "assets" / f"{name}.stl"))

    drift, expected = [], []
    for ours, theirs in pairs:
        if not theirs.exists():
            drift.append(f"{ours.name}: no counterpart at {theirs}")
        elif ours.read_bytes() != theirs.read_bytes():
            drift.append(f"{ours.name}: differs from {theirs}")

    # The arm XML is compared joint-range by joint-range so an intentional
    # limit change doesn't mask an unrelated edit elsewhere in the file.
    if not their_arm.exists():
        drift.append(f"yam_ultra.xml: no counterpart at {their_arm}")
    else:
        ours_r, theirs_r = _joint_ranges(ARM_XML), _joint_ranges(their_arm)
        for name in sorted(set(ours_r) | set(theirs_r)):
            o, t = ours_r.get(name), theirs_r.get(name)
            if o is None or t is None:
                drift.append(f"joint {name}: present in only one model ({o=}, {t=})")
                continue
            if all(abs(a - b) < 1e-6 for a, b in zip(o, t)):
                continue
            want = INTENTIONAL_DEVIATIONS.get(name)
            if want and all(abs(a - b) < 1e-6 for a, b in zip(o, want[0])):
                expected.append(f"joint {name}: {o} vs i2rt {t} — {want[1]}")
            else:
                drift.append(f"joint {name}: {o} vs i2rt {t}"
                             + (f" (expected {want[0]})" if want else ""))
        if ARM_XML.read_bytes() != their_arm.read_bytes() and not (
                set(ours_r.items()) ^ set(theirs_r.items())):
            drift.append("yam_ultra.xml: differs from i2rt outside the joint ranges")

    for e in expected:
        print(f"expected deviation — {e}")
    if drift:
        print(f"WARNING: {len(drift)} unexpected difference(s) from i2rt:")
        for d in drift:
            print(f"  - {d}")
        print("  If unintended, re-copy from i2rt (see the module docstring).")
    else:
        print(f"vendored model matches i2rt "
              f"({len(pairs)} files + all joint ranges checked, "
              f"{len(expected)} intentional deviation(s))")


def _self_test() -> None:
    """Compile the combined model and verify the wrist geometry and the
    3+3 decoupling assumption that ik/decoupled_ik.py relies on."""
    _check_against_i2rt()
    model, data = build_model_with_tool0_site()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool0")
    anchor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "j4_anchor")
    assert -1 not in (site_id, anchor_id), "required site missing"
    print(f"compiled: nq={model.nq} (6 arm + 2 coupled finger slides)")

    data.qpos[:6] = DEFAULT_Q_REST
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    print(f"tool0 at rest: {data.site_xpos[site_id]}")

    # Wrist axes must be independent (the stub arm XML alone would give
    # det = 0 with joints 5/6 coaxial; the mount transform fixes that).
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    det_rot = float(np.linalg.det(jacr[:, 3:6]))
    mujoco.mj_jacSite(model, data, jacp, jacr, anchor_id)
    det_pos = float(np.linalg.det(jacp[:, :3]))
    print(f"det(J_rot wrist) = {det_rot:.4f}   det(J_pos joints1-3) = {det_pos:.4f}")
    assert abs(det_rot) > 0.5, "wrist axes not independent — mount transform broken?"
    assert abs(det_pos) > 1e-3, "rest pose is position-singular"

    # Joints 5 and 6 must intersect (2-axis wrist core); joint 4 sits a
    # known offset upstream — report the non-sphericity.
    def joint_line(j: int) -> tuple[np.ndarray, np.ndarray]:
        bid = model.jnt_bodyid[j]
        R = data.xmat[bid].reshape(3, 3)
        return data.xpos[bid] + R @ model.jnt_pos[j], R @ model.jnt_axis[j]

    p5, a5 = joint_line(4)
    p6, a6 = joint_line(5)
    cross = np.cross(a5, a6)
    gap56 = abs(float((p6 - p5) @ cross / np.linalg.norm(cross)))
    # Wrist center = the j5/j6 axes' (near-)intersection point.
    t5 = float(np.cross(p6 - p5, a6) @ cross / (cross @ cross))
    wrist_center = p5 + t5 * a5
    p4, a4 = joint_line(3)
    v = wrist_center - p4
    off4 = float(np.linalg.norm(v - (v @ a4) * a4))
    print(f"joint5/6 axis gap = {gap56 * 1000:.2f} mm   "
          f"joint4 offset from wrist center = {off4 * 1000:.1f} mm")
    assert gap56 < 1e-3, "joints 5/6 do not intersect"


if __name__ == "__main__":
    _self_test()
