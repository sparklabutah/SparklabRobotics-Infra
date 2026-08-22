"""Bimanual i2rt YAM-Ultra: two 6-DoF arms + LINEAR_4310 grippers on CAN,
three RealSense cameras, driven by Quest teleoperation or a LeRobot policy.

    --robot.type=yam_ultra_bimanual        real arms on CAN
    --robot.type=yam_ultra_sim             the Isaac twin, rendered cameras
    --teleop.type=bi_quest_teleop | single_arm_quest_teleop

The arms run in their own processes (``arm_server.py``), reached over portal
RPC via ``arm_client.py``. The re-exports below are required by LeRobot's
class resolution, and the name deferral below handles a co-installed
predecessor distribution — see DESIGN.md for both.
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
# Unguarded on purpose: "yam_ultra_sim" is unique here, so a collision is a
# genuine conflict. Imports nothing from follower.py, so it survives the skip.
from .sim_follower import YamUltraSim, YamUltraSimConfig  # noqa: F401,E402

# --- teleoperators ----------------------------------------------------------
# Do NOT pre-bind `teleop = None` above this: `from . import teleop` only imports
# the submodule when the parent lacks the attribute, so it would silently no-op.
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
