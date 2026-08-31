"""Drive both physical YAM-Ultra arms with Quest teleop, with arm/disarm.

One BiQuestTeleoperator driving a YamUltraFollower. Action keys `left_*` go to
--left-channel, `right_*` to --right-channel; swap the flags, not the
controllers, on a mirrored rig. Cameras are disabled here — the relay already
owns the RealSense devices, and two processes cannot share a serial.

The startup pose of each arm is captured as HOME; every automatic motion
targets it and nothing ever raises an arm on its own. Starts DISARMED. B/Y on
either hand arms it, seeded from the measured pose; B/Y again disarms and ramps
both arms home. Grip clutches, trigger drives the gripper, A/X is precision,
thumbstick ramps that arm home. Ctrl-C ramps home then torques off; --no-park
skips the ramp::

    python -m lerobot_robot_sparklab.robots.yam_ultra.teleop_bimanual \
        --ws-url wss://127.0.0.1:8443/ws        # add --sim for no hardware
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from lerobot_robot_sparklab.robots.yam_ultra.teleop.bi_quest_teleop import (
    BiQuestTeleoperator,
    BiQuestTeleoperatorConfig,
)
from lerobot_robot_sparklab.robots.yam_ultra.follower import (
    ARM_JOINTS,
    HANDS,
    YamUltraFollower,
    YamUltraFollowerConfig,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
)
logger = logging.getLogger("teleop_bimanual")


def seed_from_follower(teleop: BiQuestTeleoperator, follower: YamUltraFollower) -> None:
    teleop.seed_qpos_from_obs(follower.get_observation())


def main() -> None:
    # A background job inherits SIGINT ignored, so Python never installs
    # KeyboardInterrupt; restore it so Ctrl-C always reaches the safe shutdown.
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--left-channel", default="can_left")
    ap.add_argument("--right-channel", default="can_right")
    ap.add_argument("--ws-url", default="wss://127.0.0.1:8443/ws")
    ap.add_argument("--freq", type=int, default=100, help="control loop rate (Hz)")
    ap.add_argument("--sim", action="store_true",
                    help="use i2rt SimRobots instead of CAN hardware")
    ap.add_argument("--gripper-flip", action="store_true",
                    help="invert the trigger→gripper mapping on both arms")
    ap.add_argument("--home-ramp-s", type=float, default=5.0,
                    help="duration of the disarm / exit go-home ramps")
    ap.add_argument("--rest-ramp-s", type=float, default=4.0,
                    help="duration of the thumbstick go-home ramp (teleop-side)")
    ap.add_argument("--max-dq-pos", type=float, default=0.04)
    ap.add_argument("--max-dq-rot", type=float, default=0.16)
    ap.add_argument("--scale-translation", type=float, default=0.8)
    ap.add_argument("--scale-rotation", type=float, default=0.8)
    ap.add_argument("--no-park", action="store_true",
                    help="on exit, do not ramp home before torques off")
    ap.add_argument("--id", default="vr-teleop-yam-bi-hw")
    args = ap.parse_args()

    logger.info("connecting both arms ... (gripper will self-home, keep clear)")
    follower = YamUltraFollower(YamUltraFollowerConfig(
        left_channel=args.left_channel,
        right_channel=args.right_channel,
        sim=args.sim,
        gripper_flip=args.gripper_flip,
        max_relative_target=[args.max_dq_pos] * 3 + [args.max_dq_rot] * 3,
        # This script parks explicitly in its finally block (and on disarm), so
        # don't ramp a second time inside disconnect(). --no-park disables both.
        park_on_disconnect=False,
        cameras={},  # relay owns the RealSense devices for the VR view
    ))
    follower.connect()
    n_dofs = follower.n_dofs

    # The startup pose IS the home pose: disarm/thumbstick/exit all
    # return here, and the IK's posture bias pulls toward it.
    obs0 = follower.get_observation()
    home = {h: np.array([obs0[f"{h}_joint_{j}.pos"] for j in range(1, ARM_JOINTS + 1)])
            for h in HANDS}
    for h in HANDS:
        logger.info("%s home pose (captured at startup): %s", h, np.round(home[h], 3).tolist())

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
        id=args.id,
        ws_url=args.ws_url,
        rest_qpos_left=home["left"].tolist(),
        rest_qpos_right=home["right"].tolist(),
        max_dq_per_joint=[args.max_dq_pos] * 3 + [args.max_dq_rot] * 3,
        max_dq_per_joint_scalar_pos=args.max_dq_pos,
        max_dq_per_joint_scalar_rot=args.max_dq_rot,
        scale_translation=args.scale_translation,
        scale_rotation=args.scale_rotation,
        rest_ramp_duration_s=args.rest_ramp_s,
    ))
    teleop.connect()
    seed_from_follower(teleop, follower)

    armed = False
    last_handoff = False
    cmd_gripper = {h: obs0[f"{h}_gripper.pos"] for h in HANDS}
    logger.info("DISARMED, holding home — press B/Y to arm. Ctrl-C ramps home & exits.")

    period = 1.0 / args.freq
    next_tick = time.perf_counter()
    try:
        while True:
            action = teleop.get_action()

            # B/Y rising edge toggles ARMED <-> DISARMED.
            handoff = teleop.is_handoff_pressed()
            if handoff and not last_handoff:
                last_handoff = handoff
                if not armed:
                    seed_from_follower(teleop, follower)
                    armed = True
                    logger.info("ARMED — grip to engage; the arm follows from where it is. "
                                "B/Y to disarm.")
                else:
                    armed = False
                    logger.info("DISARMING: ramping both arms home (%.1fs) ...",
                                args.home_ramp_s)
                    follower.move_to(home, args.home_ramp_s)
                    # Re-sync the teleop to the parked pose so its internal
                    # state (and the viewer) never diverges from the robots.
                    seed_from_follower(teleop, follower)
                    logger.info("DISARMED, holding home — B/Y to re-arm.")
                # Refresh the held gripper command from the measured state.
                obs = follower.get_observation()
                for h in HANDS:
                    cmd_gripper[h] = obs[f"{h}_gripper.pos"]
                next_tick = time.perf_counter()
                # The action in hand predates the seed — never command it.
                continue
            last_handoff = handoff

            if armed:
                send: dict[str, float] = {}
                for h in HANDS:
                    for j in range(1, ARM_JOINTS + 1):
                        send[f"{h}_joint_{j}.pos"] = float(action[f"{h}_joint_{j}.pos"])
                    if n_dofs[h] > ARM_JOINTS:
                        cmd_gripper[h] = float(action.get(f"{h}_gripper.pos", 0.0))
                    send[f"{h}_gripper.pos"] = cmd_gripper[h]
                follower.send_action(send)

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
                logger.info("ramping home before torques off ...")
                follower.move_to(home, args.home_ramp_s)
        finally:
            follower.disconnect()


if __name__ == "__main__":
    main()
