"""Bimanual i2rt YAM-Ultra: two 6-DoF arms + LINEAR_4310 grippers on CAN,
three RealSense cameras, driven by Quest teleoperation or a LeRobot policy.

    --robot.type=yam_ultra_bimanual        real arms on CAN
    --robot.type=yam_ultra_sim             the Isaac twin, rendered cameras
    --teleop.type=bi_quest_teleop | single_arm_quest_teleop

The arms run in their own processes (``arm_server.py``), reached over portal
RPC via ``arm_client.py``. These re-exports trigger LeRobot registration.
"""

from . import teleop  # noqa: F401
from .follower import YamUltraFollower, YamUltraFollowerConfig  # noqa: F401
from .sim_follower import YamUltraSim, YamUltraSimConfig  # noqa: F401

__all__ = ["YamUltraFollower", "YamUltraFollowerConfig",
           "YamUltraSim", "YamUltraSimConfig", "teleop"]
