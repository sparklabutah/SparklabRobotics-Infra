"""Bimanual i2rt YAM-Ultra.

Two 6-DoF arms + LINEAR_4310 grippers on CAN, three RealSense cameras,
driven either by Quest teleoperation or a LeRobot policy.

    --robot.type=yam_ultra_bimanual
    --teleop.type=bi_quest_teleop | single_arm_quest_teleop

The arms run in their own processes (``arm_server.py``); this package talks
to them over portal RPC via ``arm_client.py``. See ``arm_server``'s docstring
for why — in-process, i2rt's CAN thread shared a GIL with policy inference
and got starved into `loss communication`.

The re-export below is required, not cosmetic: LeRobot's
``make_device_from_device_class`` resolves a Robot class from the **direct
parent package** of its Config's module and never walks further up, so
``YamUltraFollower`` must be importable from exactly here.
"""

from .follower import YamUltraFollower, YamUltraFollowerConfig  # noqa: F401

# Import side effect: registers bi_quest_teleop / single_arm_quest_teleop.
from . import teleop  # noqa: F401

__all__ = ["YamUltraFollower", "YamUltraFollowerConfig", "teleop"]
