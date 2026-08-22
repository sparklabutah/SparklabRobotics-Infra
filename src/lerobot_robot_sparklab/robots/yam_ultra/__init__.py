"""Bimanual i2rt YAM-Ultra: two 6-DoF arms + LINEAR_4310 grippers on CAN,
three RealSense cameras, driven by Quest teleoperation or a LeRobot policy.

    --robot.type=yam_ultra_bimanual        real arms on CAN
    --robot.type=yam_ultra_sim             the Isaac twin, rendered cameras
    --teleop.type=bi_quest_teleop | single_arm_quest_teleop

The arms run in their own processes (``arm_server.py``), reached over portal
RPC via ``arm_client.py``.

The re-exports below are required: LeRobot's ``make_device_from_device_class``
resolves a Robot class from the **direct parent package** of its Config's
module and never walks further up.

DEFERRAL. Every ``lerobot-*`` CLI imports every installed ``lerobot_robot_*``
distribution, so while the pre-restructure package (``lerobot_robot_yam_ultra``,
../SparkRobot) is also installed both claim the same three names. LeRobot's
registry raises on whichever import happens to run second, and that import
aborts partway — silently dropping whatever it had not yet registered. Rather
than let import order decide, this package skips the shared names entirely
whenever the predecessor is installed, so the hardware-tested code always wins
them. Uninstall the predecessor and the skip stops on its own.
"""

import importlib.metadata as _md
import logging

logger = logging.getLogger(__name__)

_PREDECESSOR = "lerobot_robot_yam_ultra"


def _predecessor_installed() -> bool:
    try:
        _md.distribution(_PREDECESSOR)
        return True
    except _md.PackageNotFoundError:
        return False


_DEFER = _predecessor_installed()
if _DEFER:
    logger.warning(
        "%s is installed and owns yam_ultra_bimanual / bi_quest_teleop / "
        "single_arm_quest_teleop; deferring those names to it. SparkLab still "
        "provides yam_ultra_sim.", _PREDECESSOR)


def _already_registered(exc: Exception) -> bool:
    # Narrow on purpose: any other ValueError from these imports is a real bug.
    return isinstance(exc, ValueError) and "already registered" in str(exc)


def _skipped(what: str, exc: Exception) -> None:
    logger.warning(
        "%s: another installed distribution already claims this name, so "
        "SparkLab's version is NOT active (%s).", what, exc)


# --- hardware follower ------------------------------------------------------
YamUltraFollower = YamUltraFollowerConfig = None  # type: ignore[assignment]
if not _DEFER:
    try:
        from .follower import YamUltraFollower, YamUltraFollowerConfig  # noqa: F401
    except ValueError as exc:
        if not _already_registered(exc):
            raise
        _skipped("yam_ultra_bimanual", exc)

# --- Isaac-backed simulator -------------------------------------------------
# Unguarded on purpose: "yam_ultra_sim" is unique to this distribution, so a
# collision would be a genuine conflict worth failing on. It imports nothing
# from follower.py, so it survives the follower being skipped above.
from .sim_follower import YamUltraSim, YamUltraSimConfig  # noqa: F401,E402

# --- teleoperators ----------------------------------------------------------
# Registers bi_quest_teleop / single_arm_quest_teleop as an import side effect.
#
# Do NOT pre-bind `teleop = None` above this. `from . import teleop` imports the
# submodule only when the parent lacks that attribute (CPython's
# _handle_fromlist: `elif not hasattr(module, x)`), so a placeholder makes the
# import a silent no-op — nothing raises, the teleoperators just never register.
if not _DEFER:
    try:
        from . import teleop  # noqa: F401
    except ValueError as exc:
        if not _already_registered(exc):
            raise
        _skipped("bi_quest_teleop / single_arm_quest_teleop", exc)
        teleop = None  # type: ignore[assignment]
else:
    teleop = None  # type: ignore[assignment]

__all__ = ["YamUltraFollower", "YamUltraFollowerConfig",
           "YamUltraSim", "YamUltraSimConfig", "teleop"]
