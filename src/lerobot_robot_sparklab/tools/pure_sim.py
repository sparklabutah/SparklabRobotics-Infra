"""Pure-simulation VR teleop — no hardware follower.

Runs `BiQuestTeleoperator` against the Quest pose stream and publishes
`ik_state` back to the relay instead of commanding motors. The same IK pipeline
as `teleop_bimanual.py`, minus the follower and CAN handling::

    sparklab-relay                                    # 1. relay
    python -m lerobot_robot_sparklab.tools.pure_sim   # 2. this loop
    # 3. Quest browser -> the relay page -> Start Teleop -> squeeze a grip

To watch it on the workstation: ./scripts/isaac_python.sh -m sparklab_sim.live
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from lerobot.utils.utils import init_logging

from lerobot_robot_sparklab.robots.yam_ultra.teleop.bi_quest_teleop import (
    BiQuestTeleoperator,
    BiQuestTeleoperatorConfig,
)
from lerobot_robot_sparklab.robots.yam_ultra.teleop.cli import (
    add_ik_cli_args,
    ik_kwargs_from_args,
    parse_rest_pose_env,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws")
    ap.add_argument("--freq", type=int, default=200, help="IK loop rate (Hz)")
    ap.add_argument("--id", default="vr-teleop-sim",
                    help="teleop id stamped into ik_state broadcasts, so a "
                         "listener can pick one stream when several sims share "
                         "a relay")
    add_ik_cli_args(ap)
    args = ap.parse_args()

    init_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fallback = [0.0, float(np.pi / 2), float(np.pi / 2), 0.0, 0.0, 0.0]
    rest_left = parse_rest_pose_env("LEFT_REST_POSE", fallback)
    rest_right = parse_rest_pose_env("RIGHT_REST_POSE", fallback)

    ik_overrides = ik_kwargs_from_args(args)
    logger.info("======== pure-sim vr-teleop config ========")
    logger.info("ws_url       : %s", args.ws_url)
    logger.info("freq         : %d Hz", args.freq)
    logger.info("rest         : left=%s  right=%s",
                [round(x, 3) for x in rest_left],
                [round(x, 3) for x in rest_right])
    if ik_overrides:
        logger.info("CLI IK overrides: %s", ik_overrides)
    logger.info("==========================================")

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
        id=args.id,
        ws_url=args.ws_url,
        rest_qpos_left=rest_left,
        rest_qpos_right=rest_right,
        **ik_overrides,
    ))

    teleop.connect()
    logger.info("pure-sim loop running at %d Hz — Ctrl-C to stop", args.freq)

    period = 1.0 / args.freq
    next_tick = time.perf_counter()
    try:
        while True:
            # get_action() drives one IK tick and (with publish_ik_state=True)
            # broadcasts the resulting qpos back to the relay for the viewer.
            teleop.get_action()
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


if __name__ == "__main__":
    main()
