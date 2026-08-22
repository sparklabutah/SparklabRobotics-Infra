"""Drive a physical YAM-Ultra with Quest teleop — single arm.

Bridges relay → SingleArmQuestTeleoperator → IK to an i2rt MotorChainRobot over
CAN: each tick the action dict becomes a 7-vector [joint1..6 rad, gripper 0..1].

The arm does not move at startup — the teleop is seeded from the measured
joint positions, though the gripper self-homes once during driver init. Click
the thumbstick to ramp to the rest pose, then squeeze the grip to clutch in.
Δq caps default to half the sim values. Ctrl-C parks to the folded zero pose
and torques off; --no-park torques off where it stands::

    python -m lerobot_robot_sparklab.robots.yam_ultra.teleop_single \
        --channel can_right --ws-url wss://127.0.0.1:8443/ws   # --sim to dry-run
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from lerobot_robot_sparklab.robots.yam_ultra.connection import connect_yam_arm
from lerobot_robot_sparklab.robots.yam_ultra.teleop.single_arm_quest_teleop import (
    SingleArmQuestTeleoperator,
    SingleArmQuestTeleoperatorConfig,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
)
logger = logging.getLogger("teleop_single")

ARM_JOINTS = 6


def park_ramp(robot, duration_s: float = 5.0, rate_hz: float = 50.0) -> None:
    """Slow interpolated move of the arm joints to the folded zero pose.

    A plain command_joint_pos ramp, since SimRobot has no move_joints.
    """
    cur = np.asarray(robot.get_joint_pos(), dtype=float)
    park = cur.copy()
    park[:ARM_JOINTS] = 0.0  # folded resting pose; gripper stays put
    steps = max(1, int(duration_s * rate_hz))
    for i in range(steps + 1):
        a = i / steps
        robot.command_joint_pos((1.0 - a) * cur + a * park)
        time.sleep(1.0 / rate_hz)


def main() -> None:
    # A background job inherits SIGINT ignored; the safe-park shutdown needs it.
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--channel", default="can_right",
                    help="CAN interface of the arm (this machine: can_right / can_left)")
    ap.add_argument("--arm", choices=("left", "right"), default="right",
                    help="which Quest hand drives the arm")
    ap.add_argument("--ws-url", default="wss://127.0.0.1:8443/ws")
    ap.add_argument("--freq", type=int, default=100, help="control loop rate (Hz)")
    ap.add_argument("--sim", action="store_true",
                    help="use i2rt's SimRobot instead of CAN hardware (bridge dry-run)")
    ap.add_argument("--gripper-flip", action="store_true",
                    help="invert the trigger→gripper mapping (use if trigger opens instead of closes)")
    ap.add_argument("--rest-ramp-s", type=float, default=4.0,
                    help="duration of the thumbstick go-to-rest ramp")
    ap.add_argument("--max-dq-pos", type=float, default=0.04,
                    help="per-tick Δq cap, joints 1-3 (rad/tick)")
    ap.add_argument("--max-dq-rot", type=float, default=0.16,
                    help="per-tick Δq cap, joints 4-6 (rad/tick)")
    ap.add_argument("--scale-translation", type=float, default=0.8,
                    help="controller→EE translation gain (repo default 1.5; "
                         "the web Settings panel overrides this live)")
    ap.add_argument("--scale-rotation", type=float, default=0.8,
                    help="controller→EE rotation gain (repo default 1.5; "
                         "the web Settings panel overrides this live)")
    ap.add_argument("--no-park", action="store_true",
                    help="on exit, do not move to the folded pose before torques off")
    ap.add_argument("--id", default="vr-teleop-yam-hw")
    args = ap.parse_args()

    logger.info("connecting to YAM-Ultra on %s%s ...", args.channel, " (SIM)" if args.sim else "")
    logger.info("note: the gripper self-homes during init — keep clear.")
    robot = connect_yam_arm(args.channel, sim=args.sim)
    n_dofs = robot.num_dofs()

    teleop = SingleArmQuestTeleoperator(SingleArmQuestTeleoperatorConfig(
        id=args.id,
        arm=args.arm,
        ws_url=args.ws_url,
        max_dq_per_joint=[args.max_dq_pos] * 3 + [args.max_dq_rot] * 3,
        max_dq_per_joint_scalar_pos=args.max_dq_pos,
        max_dq_per_joint_scalar_rot=args.max_dq_rot,
        scale_translation=args.scale_translation,
        scale_rotation=args.scale_rotation,
        rest_ramp_duration_s=args.rest_ramp_s,
    ))
    teleop.connect()

    # Seed the teleop with the robot's measured pose so the first emitted
    # actions hold position instead of snapping to the teleop's rest pose.
    q0 = np.asarray(robot.get_joint_pos(), dtype=float)
    obs = {f"joint_{j}.pos": float(q0[j - 1]) for j in range(1, ARM_JOINTS + 1)}
    if n_dofs > ARM_JOINTS:
        g0 = float(q0[ARM_JOINTS])
        obs["gripper.pos"] = 1.0 - g0 if args.gripper_flip else g0
    teleop.seed_qpos_from_obs(obs)
    logger.info("seeded teleop from robot pose: %s", np.round(q0, 3).tolist())
    logger.info("loop at %d Hz — thumbstick = go to rest, grip = engage, Ctrl-C = park & exit",
                args.freq)

    cmd = q0.copy()
    period = 1.0 / args.freq
    next_tick = time.perf_counter()
    try:
        while True:
            action = teleop.get_action()
            for j in range(1, ARM_JOINTS + 1):
                cmd[j - 1] = action[f"joint_{j}.pos"]
            if n_dofs > ARM_JOINTS:
                g = float(action.get("gripper.pos", 0.0))
                cmd[ARM_JOINTS] = 1.0 - g if args.gripper_flip else g
            robot.command_joint_pos(cmd)
            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        teleop.disconnect()
        try:
            if not args.no_park:
                logger.info("parking (slow move to folded pose) ...")
                park_ramp(robot, duration_s=5.0)
        finally:
            robot.close()


if __name__ == "__main__":
    main()
