"""MuJoCo model construction for the I2RT YAM-Ultra arm.

Rebuilds the two named sites the IK pipeline needs from the YAM-Ultra's own
MJCF, and numerically re-checks the 3+3 joint-decoupling assumption
``ik/decoupled_ik.py`` relies on (see ``_self_test``). The IK math itself lives
in ``ik/decoupled_ik.py``.

VENDORED MODELS, copied byte-for-byte from i2rt:

  yam_ultra.xml, assets/{base,link1..link5}.stl
      <i2rt>/i2rt/robot_models/arm/yam_ultra/
  linear_4310.xml, assets/{gripper,tip_left,tip_right}.stl
      <i2rt>/i2rt/robot_models/gripper/linear_4310/

The arm's ``link6`` body is a stub; upstream i2rt sets the real wrist-mount
transform and grafts the gripper subtree at runtime.
``combine_arm_and_gripper_xml()`` reproduces that step self-containedly. Without
the mount transform joint6 would sit coaxial with joint5 and the wrist would be
degenerate.

JOINT LIMITS. Everything matches i2rt except ``INTENTIONAL_DEVIATIONS`` —
currently joint3's upper limit, 3.0 rad here against i2rt's nominal π. **Not
drift; do not "fix" it back.** i2rt's XML is nominal for the arm family; this
rig's elbow stops around 3.0 and commanding past that grinds it into the stop.

Getting limits wrong is quiet and nasty both ways, because the IK reads them
straight off this model and both clamps into them and derives the limit-pressure
haptic from them: too tight gives an invisible wall short of real travel, too
loose drives into a mechanical stop believing there is room. Re-run
``_check_against_i2rt()`` after any i2rt upgrade —

    python -m lerobot_robot_sparklab.robots.yam_ultra.model.kinematics

Synced from i2rt v1.2.4, whose frame-alignment pass replaced the hand-simplified
body frames with the URDF's native ones — which is why ``j4_anchor`` and
``LINK6_MOUNT_POS``/``LINK6_MOUNT_QUAT`` had to be recomputed rather than
copied. That release also fixed an upstream bug that had joint2/joint3's upper
limits swapped, so the ~0.14 rad gap between our 3.0 and π is measured against
a correct target for the first time — re-measure the physical stop before
assuming it is still real.

Meshes never affect kinematics (MuJoCo builds FK/Jacobians from body frames and
joint axes) and gravity compensation runs on i2rt's model, not this one — but
keep them in sync so the viewer shows the arm that is actually moving.

SITES added to the compiled model:

  tool0      — the EE target, at the gripper's grasp center (the same pose as
               the gripper XML's ``grasp_site``, which i2rt's own IK uses).
  j4_anchor  — the position-task anchor for the decoupled IK. On link3
               (upstream of joint 4, so wrist-invariant), 10 cm past joint 4's
               pivot along link3→link4, which keeps it off the joint-1 axis at
               folded poses.

Decoupling assumption (checked numerically): joints 1-3 are yaw + two parallel
pitches (position), joints 4-6 are pitch ⊥ yaw ⊥ roll (orientation). Joints 5
and 6 intersect; joint 4 sits 7.0 cm off the wrist center — a modest
non-sphericity absorbed by the operator's visual feedback loop.
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
# URDF form of the same arm (gripper already assembled), vendored from i2rt
# v1.2.4 for USD/sim importers. Nothing in the control path loads it; the IK
# and viewer use the MJCF above.
ARM_URDF = _HERE / "yam_ultra.urdf"

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
    # Same physical stop, same value, expressed in the URDF's naming. The
    # deviation belongs to the arm, not to one file format, so both vendored
    # assets carry it and _check_against_i2rt() verifies both.
    "dof_joint3": ((0.0, 3.0),
                   "URDF counterpart of the joint3 deviation above"),
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
    model before compiling (the removed ``inspect_model.py`` bolted on
    position actuators to get one slider per joint) without that ever touching
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


def _urdf_joint_ranges(urdf_path: Path) -> dict[str, tuple[float, float]]:
    """joint name -> (lower, upper) as written in a URDF file."""
    out: dict[str, tuple[float, float]] = {}
    for j in ET.parse(urdf_path).getroot().iter("joint"):
        name, lim = j.get("name"), j.find("limit")
        if name and lim is not None:
            lo, hi = lim.get("lower"), lim.get("upper")
            if lo is not None and hi is not None:
                out[name] = (float(lo), float(hi))
    return out


# Fingerprint of i2rt's PRE-v1.2.4 URDF, where joint2 and joint3's limits were
# swapped. Our URDF is vendored from v1.2.4, so an older installed i2rt will
# differ on every joint — reporting that as drift would be technically true and
# useless noise. Detect the old file and say so instead.
_PRE_V124_JOINT3_UPPER = 3.66519


def _check_vendored_assets_agree() -> int:
    """Check the vendored MJCF and URDF describe the same arm. Returns the
    number of disagreements found (0 = good).

    Deliberately independent of i2rt: these two files are both ours, and the
    reason the joint3 deviation is applied to each of them is so they stay
    consistent. That invariant should be checkable on a machine that has
    never had i2rt installed.
    """
    mjcf, urdf = _joint_ranges(ARM_XML), _urdf_joint_ranges(ARM_URDF)
    problems: list[str] = []
    for i in range(1, 7):
        mj, ur = f"joint{i}", f"dof_joint{i}"
        o, u = mjcf.get(mj), urdf.get(ur)
        if o is None or u is None:
            problems.append(f"{mj}/{ur}: missing from one vendored asset ({o=}, {u=})")
        elif not all(abs(a - b) < 1e-6 for a, b in zip(o, u)):
            problems.append(f"{mj} {o} (MJCF) disagrees with {ur} {u} (URDF)")

    if problems:
        print(f"WARNING: vendored MJCF and URDF disagree ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        print("  Both must carry the same limits — see INTENTIONAL_DEVIATIONS.")
    else:
        print("vendored MJCF and URDF agree on all 6 joint ranges")
    return len(problems)


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

    # Both vendored assets came from v1.2.4, which regenerated the meshes and
    # fixed the swapped joint2/joint3 limits. Against an older installed i2rt
    # essentially everything differs — reporting each one as drift is true but
    # useless, and a check that always warns is a check nobody reads. Detect
    # the old release once and compare only what is still comparable.
    their_predates_v124 = False
    if their_arm.exists():
        j3 = _joint_ranges(their_arm).get("joint3")
        their_predates_v124 = bool(j3 and abs(j3[1] - _PRE_V124_JOINT3_UPPER) < 1e-6)

    # Meshes + the gripper XML carry no intentional deviations: byte-compare.
    pairs = [(GRIPPER_XML, root / "gripper" / "linear_4310" / "linear_4310.xml")]
    for name in ("base", "link1", "link2", "link3", "link4", "link5"):
        pairs.append((_HERE / "assets" / f"{name}.stl",
                      root / "arm" / "yam_ultra" / "assets" / f"{name}.stl"))
    for name in ("gripper", "tip_left", "tip_right"):
        pairs.append((_HERE / "assets" / f"{name}.stl",
                      root / "gripper" / "linear_4310" / "assets" / f"{name}.stl"))

    drift, expected = [], []
    if their_predates_v124:
        expected.append(
            "installed i2rt predates v1.2.4 — it regenerated the meshes and fixed "
            "the swapped joint2/joint3 limits, so the vendored v1.2.4 assets are "
            "EXPECTED to differ throughout. Skipping mesh and joint-range "
            "comparisons; re-run after upgrading i2rt to make this check useful "
            "again.")
    else:
        for ours, theirs in pairs:
            if not theirs.exists():
                drift.append(f"{ours.name}: no counterpart at {theirs}")
            elif ours.read_bytes() != theirs.read_bytes():
                drift.append(f"{ours.name}: differs from {theirs}")

    # The arm XML is compared joint-range by joint-range so an intentional
    # limit change doesn't mask an unrelated edit elsewhere in the file.
    if their_predates_v124:
        pass
    elif not their_arm.exists():
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

    # Same treatment for the vendored URDF. Compared by joint range only: it
    # carries the assembled gripper and its own naming (dof_joint*), so a
    # byte-compare would say "differs" without saying anything useful.
    their_urdf = root / "arm" / "yam_ultra" / "yam_ultra.urdf"
    if their_predates_v124:
        pass  # already reported once above, for both assets
    elif not their_urdf.exists():
        drift.append(f"yam_ultra.urdf: no counterpart at {their_urdf}")
    else:
        ours_u, theirs_u = _urdf_joint_ranges(ARM_URDF), _urdf_joint_ranges(their_urdf)
        for name in sorted(set(ours_u) | set(theirs_u)):
            o, t = ours_u.get(name), theirs_u.get(name)
            if o is None or t is None:
                drift.append(f"urdf joint {name}: present in only one model ({o=}, {t=})")
                continue
            if all(abs(a - b) < 1e-6 for a, b in zip(o, t)):
                continue
            want = INTENTIONAL_DEVIATIONS.get(name)
            if want and all(abs(a - b) < 1e-6 for a, b in zip(o, want[0])):
                expected.append(f"urdf joint {name}: {o} vs i2rt {t} — {want[1]}")
            else:
                drift.append(f"urdf joint {name}: {o} vs i2rt {t}"
                             + (f" (expected {want[0]})" if want else ""))

    for e in expected:
        print(f"expected deviation — {e}")
    if drift:
        print(f"WARNING: {len(drift)} unexpected difference(s) from i2rt:")
        for d in drift:
            print(f"  - {d}")
        print("  If unintended, re-copy from i2rt (see the module docstring).")
    elif their_predates_v124:
        # Nothing was actually compared, so don't claim a match.
        print("vendored model NOT verified against i2rt — install predates v1.2.4 "
              "(see above). The MJCF/URDF agreement check above still applies.")
    else:
        print(f"vendored model matches i2rt "
              f"({len(pairs)} files + all joint ranges checked, "
              f"{len(expected)} intentional deviation(s))")


def _self_test() -> None:
    """Compile the combined model and verify the wrist geometry and the
    3+3 decoupling assumption that ik/decoupled_ik.py relies on."""
    _check_vendored_assets_agree()
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
