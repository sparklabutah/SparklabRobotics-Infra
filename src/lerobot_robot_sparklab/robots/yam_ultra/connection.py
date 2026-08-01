"""Shared i2rt connection helper.

The one place that calls ``get_yam_robot`` with this rig's arm/gripper
type, so ``teleop_single.py``, ``teleop_bimanual.py`` and ``follower.py``
don't each hardcode the same five keyword arguments.
"""

from __future__ import annotations

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


def connect_yam_arm(channel: str, sim: bool = False, enable_auto_recovery: bool = False):
    """Connect one YAM-Ultra arm + LINEAR_4310 gripper on ``channel``.

    The gripper self-homes during this call — keep fingers clear.

    ``enable_auto_recovery`` lets i2rt clear + re-enable an errored motor
    from inside its control loop rather than killing the loop on the first
    error (see YamUltraFollowerConfig.enable_auto_recovery).
    """
    return get_yam_robot(
        channel=channel,
        arm_type=ArmType.YAM_ULTRA,
        gripper_type=GripperType.LINEAR_4310,
        zero_gravity_mode=False,
        sim=sim,
        enable_auto_recovery=enable_auto_recovery,
    )
