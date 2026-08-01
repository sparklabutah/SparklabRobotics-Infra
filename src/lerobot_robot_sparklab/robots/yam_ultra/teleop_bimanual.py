"""Drive both physical YAM-Ultra arms with Quest teleop, with arm/disarm.

One BiQuestTeleoperator (single IK loop, both hands) driving a
YamUltraFollower (the same LeRobot Robot adapter `lerobot-record` uses —
see follower.py). Action keys `left_*` go to --left-channel,
`right_*` to --right-channel; if your rig is wired mirrored, swap the
channel flags rather than the controllers.

Cameras are disabled on the Follower here (`cameras={}`): the relay
already owns the physical RealSense devices for the live VR view
(cameras/config/cameras.yaml via run.sh), and two processes can't open the
same RealSense serial at once.

The pose each arm is in at startup (folded, resting) is captured as its
HOME pose — every automatic motion targets that pose and nothing ever
raises the arm on its own:

  * The bridge starts DISARMED, holding home.
  * B/Y (either hand) → ARM: the teleop is seeded from the robots'
    measured pose and control goes live immediately — no motion until
    you squeeze a grip and move. (grip = clutch, trigger = gripper,
    A/X = precision, thumbstick = ramp that arm back to home.)
  * B/Y again → DISARM: both arms automatically ramp back to home and
    hold; controller input is ignored until re-armed.

Ctrl-C: ramps home if needed, then torques off (home is the mechanical
resting pose, so torques-off there is safe). --no-park skips the ramp.

Run (relay + viewers up first):
    python -m lerobot_robot_sparklab.robots.yam_ultra.teleop_bimanual --ws-url wss://127.0.0.1:8443/ws

Dry-run without hardware:  add --sim
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


def ramp(follower: YamUltraFollower, targets: dict, duration_s: float, rate_hz: float = 50.0) -> None:
    """Slow synchronized interpolation of both arms' joints to per-hand
    target poses (gripper held at its last observed value)."""
    obs = follower.get_observation()
    starts = {h: np.array([obs[f"{h}_joint_{j}.pos"] for j in range(1, ARM_JOINTS + 1)])
              for h in HANDS}
    grippers = {h: obs[f"{h}_gripper.pos"] for h in HANDS}
    steps = max(1, int(duration_s * rate_hz))
    for i in range(steps + 1):
        a = i / steps
        action: dict[str, float] = {}
        for h in HANDS:
            q = (1.0 - a) * starts[h] + a * targets[h]
            for j in range(1, ARM_JOINTS + 1):
                action[f"{h}_joint_{j}.pos"] = float(q[j - 1])
            action[f"{h}_gripper.pos"] = grippers[h]
        follower.send_action(action)
        time.sleep(1.0 / rate_hz)


def seed_from_follower(teleop: BiQuestTeleoperator, follower: YamUltraFollower) -> None:
    teleop.seed_qpos_from_obs(follower.get_observation())


def main() -> None:
    # If a parent shell spawned us in the background, SIGINT arrives ignored
    # and Python never installs KeyboardInterrupt — restore it so Ctrl-C (or
    # a launcher's forwarded INT) always triggers the safe go-home shutdown.
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
        # This script's own dq_caps clamp below is the safety layer; leave both
        # of the Follower's clamps off to avoid two clamps disagreeing.
        max_relative_target=None,
        max_joint_velocity=None,
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
    cmd = {h: home[h].copy() for h in HANDS}
    cmd_gripper = {h: obs0[f"{h}_gripper.pos"] for h in HANDS}
    dq_caps = np.array([args.max_dq_pos] * 3 + [args.max_dq_rot] * 3)
    last_clamp_warn = 0.0
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
                    ramp(follower, home, args.home_ramp_s)
                    # Re-sync the teleop to the parked pose so its internal
                    # state (and the viewer) never diverges from the robots.
                    seed_from_follower(teleop, follower)
                    logger.info("DISARMED, holding home — B/Y to re-arm.")
                # Baseline for the per-tick clamp = where the arms really are.
                obs = follower.get_observation()
                for h in HANDS:
                    cmd[h] = np.array([obs[f"{h}_joint_{j}.pos"] for j in range(1, ARM_JOINTS + 1)])
                    cmd_gripper[h] = obs[f"{h}_gripper.pos"]
                next_tick = time.perf_counter()
                # The action in hand predates the seed — never command it.
                continue
            last_handoff = handoff

            if armed:
                send: dict[str, float] = {}
                for h in HANDS:
                    target = np.array([action[f"{h}_joint_{j}.pos"]
                                       for j in range(1, ARM_JOINTS + 1)])
                    # Backstop: a legit IK action never moves more than the
                    # per-tick Δq caps, so anything larger means state desync
                    # (e.g. a seeding bug) — walk toward it at cap speed
                    # instead of letting the motors snap.
                    step = target - cmd[h]
                    clamped = np.clip(step, -dq_caps, dq_caps)
                    if (np.abs(step) > dq_caps + 1e-9).any() and \
                            time.perf_counter() - last_clamp_warn > 1.0:
                        last_clamp_warn = time.perf_counter()
                        logger.warning("%s action jumped %.3f rad max — clamped to caps "
                                       "(state desync?)", h, float(np.abs(step).max()))
                    cmd[h] += clamped
                    for j in range(1, ARM_JOINTS + 1):
                        send[f"{h}_joint_{j}.pos"] = float(cmd[h][j - 1])
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
                ramp(follower, home, args.home_ramp_s)
        finally:
            follower.disconnect()


if __name__ == "__main__":
    main()
