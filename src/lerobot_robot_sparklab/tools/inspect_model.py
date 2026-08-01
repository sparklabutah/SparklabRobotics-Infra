"""Look at the IK model in MuJoCo: joint limits, reach, and the IK sites.

This is the model the teleop actually solves against — the same
``build_model_with_tool0_site()`` the ``DecoupledIKSolver`` compiles — so
whatever you see here is what the IK believes the arm can do. Use it to
confirm the limits by eye rather than trusting the numbers alone.

    # print the limit table, then open the viewer and sweep every joint
    python -m lerobot_robot_sparklab.tools.inspect_model

    # just the numbers, no window (works over SSH)
    python -m lerobot_robot_sparklab.tools.inspect_model --no-viewer

    # sweep one joint, slowly
    python -m lerobot_robot_sparklab.tools.inspect_model --joint 3 --period 6

    # hold a specific pose instead of sweeping (radians, 6 values)
    python -m lerobot_robot_sparklab.tools.inspect_model --qpos 0,1.57,1.57,0,0,0

The sweep drives each joint from its lower limit to its upper limit and
back while the others hold the rest pose, so a limit that is wrong shows
up as the arm stopping short of (or pushing past) the mechanical stop.

The two orange/─ sites the IK depends on are visible in the viewer:
``tool0`` (the EE target, at the gripper's grasp centre) and ``j4_anchor``
(the position-task anchor on link3). Turn on Model → Site in the viewer's
Rendering panel if they aren't showing.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..robots.yam_ultra.model.kinematics import (
    ARM_XML,
    DEFAULT_Q_REST,
    INTENTIONAL_DEVIATIONS,
    build_model_with_tool0_site,
)

ARM_JOINTS = 6

# i2rt's limits for the YAM-Ultra, read back from its own model so the
# comparison stays honest if i2rt is upgraded. Falls back to None when
# i2rt isn't installed (this model is usable standalone).
def _i2rt_limits() -> dict[int, tuple[float, float]] | None:
    try:
        import i2rt  # noqa: F401
    except ImportError:
        return None
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import i2rt as _i2rt
    xml = (Path(_i2rt.__file__).resolve().parent
           / "robot_models" / "arm" / "yam_ultra" / "yam_ultra.xml")
    if not xml.exists():
        return None
    out: dict[int, tuple[float, float]] = {}
    for j in ET.parse(xml).getroot().iter("joint"):
        name = j.get("name", "")
        rng = j.get("range")
        if name.startswith("joint") and rng:
            try:
                idx = int(name.removeprefix("joint"))
            except ValueError:
                continue
            lo, hi = (float(v) for v in rng.split())
            out[idx] = (lo, hi)
    return out or None


def print_limits(model) -> None:
    i2rt = _i2rt_limits()
    print(f"\nmodel: {ARM_XML.name} + gripper   nq={model.nq} "
          f"({ARM_JOINTS} arm + {model.nq - ARM_JOINTS} coupled finger slides)\n")
    head = f"{'joint':7} {'lower':>10} {'upper':>10} {'travel':>10} {'travel(deg)':>12}"
    if i2rt:
        head += f"   {'i2rt lower':>11} {'i2rt upper':>11}  match"
    print(head)
    print("-" * len(head))
    for j in range(1, ARM_JOINTS + 1):
        lo, hi = model.jnt_range[j - 1]
        row = (f"joint{j:<2} {lo:10.5f} {hi:10.5f} {hi - lo:10.5f} "
               f"{np.degrees(hi - lo):12.1f}")
        if i2rt:
            ilo, ihi = i2rt.get(j, (float('nan'),) * 2)
            ok = abs(lo - ilo) < 1e-4 and abs(hi - ihi) < 1e-4
            want = INTENTIONAL_DEVIATIONS.get(f"joint{j}")
            if ok:
                verdict = "OK"
            elif want and abs(lo - want[0][0]) < 1e-6 and abs(hi - want[0][1]) < 1e-6:
                verdict = "intended"   # measured on this rig, see model/kinematics.py
            else:
                verdict = "MISMATCH"
            row += f"   {ilo:11.5f} {ihi:11.5f}  {verdict}"
        print(row)
    # Gripper finger slides, if present.
    for j in range(ARM_JOINTS, model.njnt):
        lo, hi = model.jnt_range[j]
        print(f"joint{j + 1:<2} {lo:10.5f} {hi:10.5f} {hi - lo:10.5f} "
              f"{'':>12}   (gripper finger slide, metres)")
    if i2rt is None:
        print("\n(i2rt not installed — showing this model's limits only)")
    elif any(f"joint{j}" in INTENTIONAL_DEVIATIONS for j in range(1, ARM_JOINTS + 1)):
        print("\n'intended' = deliberately tighter than i2rt's nominal model, measured on")
        print("this rig. See INTENTIONAL_DEVIATIONS in model/kinematics.py for why:")
        for name, (_, why) in INTENTIONAL_DEVIATIONS.items():
            print(f"  {name}: {why}")
    print()


def _reach_at(model, data, qpos6, site_id) -> np.ndarray:
    import mujoco
    data.qpos[:ARM_JOINTS] = qpos6
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[site_id].copy()


def print_reach(model, data) -> None:
    """Where tool0 lands at the rest pose and at each joint's extremes —
    a quick sanity check that the limits produce sane end-effector poses."""
    import mujoco
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool0")
    rest = np.array(DEFAULT_Q_REST, dtype=float)
    print(f"tool0 at rest pose {np.round(rest, 3).tolist()}: "
          f"{np.round(_reach_at(model, data, rest, sid), 4).tolist()} m")
    print("tool0 at each joint's limits (others held at rest):")
    for j in range(ARM_JOINTS):
        lo, hi = model.jnt_range[j]
        for label, val in (("min", lo), ("max", hi)):
            q = rest.copy()
            q[j] = val
            p = _reach_at(model, data, q, sid)
            print(f"  joint{j + 1} {label} ({val:+.4f} rad, {np.degrees(val):+7.1f} deg): "
                  f"{np.round(p, 4).tolist()}")
    data.qpos[:ARM_JOINTS] = rest
    mujoco.mj_kinematics(model, data)
    print()


def sweep(model, data, joints: list[int], period: float) -> None:
    """Open the viewer and drive each joint limit-to-limit in turn."""
    import mujoco
    import mujoco.viewer

    rest = np.array(DEFAULT_Q_REST, dtype=float)
    data.qpos[:] = 0
    data.qpos[:ARM_JOINTS] = rest
    mujoco.mj_kinematics(model, data)

    print(f"sweeping joint(s) {[j + 1 for j in joints]} — close the window to stop\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Show the tool0 / j4_anchor sites without needing the UI toggle.
        viewer.opt.sitegroup[:] = 1
        viewer.sync()
        t0 = time.perf_counter()
        announced = None
        while viewer.is_running():
            elapsed = time.perf_counter() - t0
            idx = int(elapsed // period) % len(joints)
            j = joints[idx]
            lo, hi = model.jnt_range[j]
            # Triangle wave over [0,1] within this joint's slot.
            phase = (elapsed % period) / period
            a = 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)
            q = rest.copy()
            q[j] = lo + a * (hi - lo)
            if announced != j:
                announced = j
                print(f"  joint{j + 1}: {lo:+.4f} .. {hi:+.4f} rad "
                      f"({np.degrees(lo):+.1f} .. {np.degrees(hi):+.1f} deg)")
            data.qpos[:ARM_JOINTS] = q
            mujoco.mj_kinematics(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def hold(model, data, qpos6: np.ndarray) -> None:
    import mujoco
    import mujoco.viewer

    lo = model.jnt_range[:ARM_JOINTS, 0]
    hi = model.jnt_range[:ARM_JOINTS, 1]
    out = np.clip(qpos6, lo, hi)
    if not np.allclose(out, qpos6):
        for j in range(ARM_JOINTS):
            if abs(out[j] - qpos6[j]) > 1e-9:
                print(f"  joint{j + 1}: {qpos6[j]:+.4f} is outside "
                      f"[{lo[j]:+.4f}, {hi[j]:+.4f}] — clamped to {out[j]:+.4f}")
    data.qpos[:] = 0
    data.qpos[:ARM_JOINTS] = out
    mujoco.mj_kinematics(model, data)
    print(f"holding {np.round(out, 4).tolist()} — close the window to stop\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.sitegroup[:] = 1
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 30.0)


def interactive() -> None:
    """Full MuJoCo UI with one slider per joint, so you can drive each joint
    by hand and watch where it stops.

    The IK model has no actuators (the solver only ever needs FK and
    Jacobians), and MuJoCo's Control panel is built from actuators — so
    without these you would get a viewer with nothing to drag. A position
    actuator per joint, with ``ctrlrange`` pinned to that joint's range,
    turns the panel into exactly the joint sliders we want, and makes the
    slider ends coincide with the real limits.

    The actuators exist ONLY to create those sliders. We never step the
    physics: each frame copies ``ctrl`` straight into ``qpos`` and runs
    forward kinematics. That is deliberate — driving them as real position
    servos was tried and is both inexact and fragile here. The model
    carries no joint damping, so a servo soft enough to be stable
    oscillates and settles ~2 deg away from the slider value (the readout
    would simply lie about the joint angle), while stiffening it enough to
    track blows the gripper's finger slides up into NaN, since one gain
    cannot serve both radians and a 4.75 cm prismatic joint. Copying
    ctrl -> qpos makes the slider value exactly the joint angle, which is
    the entire point of a kinematic inspector.
    """
    import mujoco
    import mujoco.viewer

    from ..robots.yam_ultra.model.kinematics import build_spec_with_tool0_site

    spec = build_spec_with_tool0_site()
    # Pass 1: read each joint's range off the compiled model (the spec's
    # joint objects don't expose it as conveniently).
    ranges = {}
    probe = spec.compile()
    for j in range(probe.njnt):
        name = mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            ranges[name] = tuple(probe.jnt_range[j])

    for name, (lo, hi) in ranges.items():
        act = spec.add_actuator()
        act.name = name          # slider label; ctrl is read as a joint angle
        act.target = name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.ctrlrange = [lo, hi]
        act.ctrllimited = 1      # slider ends == the joint's real limits

    model = spec.compile()
    data = mujoco.MjData(model)

    rest = np.array(DEFAULT_Q_REST, dtype=float)
    data.qpos[:ARM_JOINTS] = rest
    for j in range(min(ARM_JOINTS, model.nu)):
        data.ctrl[j] = rest[j]
    mujoco.mj_forward(model, data)

    print("interactive viewer — open the 'Control' panel (left side) for one "
          "slider per joint.")
    print("each slider's ends ARE that joint's limits:\n")
    for j in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        lo, hi = model.actuator_ctrlrange[j]
        unit = "m" if "joint7" in (name or "") or "joint8" in (name or "") else "rad"
        extra = "" if unit == "m" else f"  ({np.degrees(lo):+.1f} .. {np.degrees(hi):+.1f} deg)"
        print(f"  {name:16} {lo:+.5f} .. {hi:+.5f} {unit}{extra}")
    print("\ntip: Rendering > Model > Site shows tool0 (grasp centre) and "
          "j4_anchor (orange, the IK position anchor).")
    print("no physics — the joint sits exactly where the slider says.\n")

    nu = model.nu
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.sitegroup[:] = 1
        while viewer.is_running():
            # Sliders drive the joints directly. mj_forward (not mj_step) =
            # kinematics only, so what you set is exactly what you get.
            data.qpos[:nu] = data.ctrl[:nu]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="full MuJoCo UI with a slider per joint, to drive each "
                         "joint by hand (gravity off, sliders clamped to the limits)")
    ap.add_argument("--no-viewer", action="store_true",
                    help="print the tables and exit (no window; works over SSH)")
    ap.add_argument("--joint", type=int, default=None, metavar="N",
                    help="sweep only joint N (1-6); default sweeps all six")
    ap.add_argument("--period", type=float, default=4.0,
                    help="seconds per joint for a full min->max->min sweep (default 4)")
    ap.add_argument("--qpos", default=None,
                    help="hold this pose instead of sweeping: 6 comma-separated radians")
    ap.add_argument("--reach", action="store_true",
                    help="also print tool0's position at every joint limit")
    args = ap.parse_args()

    model, data = build_model_with_tool0_site()
    print_limits(model)
    if args.reach:
        print_reach(model, data)

    if args.no_viewer:
        return

    if args.interactive:
        interactive()
        return

    if args.qpos:
        vals = [float(v) for v in args.qpos.split(",") if v.strip()]
        if len(vals) != ARM_JOINTS:
            raise SystemExit(f"--qpos needs {ARM_JOINTS} values, got {len(vals)}")
        hold(model, data, np.array(vals, dtype=float))
        return

    if args.joint is not None:
        if not 1 <= args.joint <= ARM_JOINTS:
            raise SystemExit(f"--joint must be 1..{ARM_JOINTS}")
        joints = [args.joint - 1]
    else:
        joints = list(range(ARM_JOINTS))
    sweep(model, data, joints, args.period)


if __name__ == "__main__":
    main()
