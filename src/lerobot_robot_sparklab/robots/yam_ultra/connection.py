"""Shared i2rt connection helper — the one place that calls ``get_yam_robot``."""

from __future__ import annotations

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


def connect_yam_arm(channel: str, sim: bool = False, enable_auto_recovery: bool = False):
    """Connect one YAM-Ultra arm + LINEAR_4310 gripper. The gripper self-homes.

    channel: CAN interface name.
    sim: kinematic mode, no CAN traffic.
    enable_auto_recovery: let i2rt clear and re-enable an errored motor from
        inside its control loop instead of killing the loop on first error.
    """
    return get_yam_robot(
        channel=channel,
        arm_type=ArmType.YAM_ULTRA,
        gripper_type=GripperType.LINEAR_4310,
        zero_gravity_mode=False,
        sim=sim,
        enable_auto_recovery=enable_auto_recovery,
    )
