"""SparkLab robotics stack — shared infrastructure plus per-robot packages.

LAYOUT
======

    relay/     FastAPI WebSocket relay + WebRTC camera streams + the WebXR
               page served to the Quest. Robot-agnostic: it moves poses and
               video, it does not know what is listening.
    quest/     Robot-agnostic Quest input — xr_frame subscription, staleness,
               xr-standard button mapping.
    core/      Clutch-relative pose mapping with absorbing reach limits.
    cameras/   RealSense tooling (calibration, advanced-mode presets,
               intrinsics export). Per-rig serials live with the robot.
    tools/     No-hardware dry-run/inspection utilities.
    robots/    One subpackage per robot. Each owns its kinematics, IK,
               hardware bridge, teleoperator and rig config.

Adding a robot: create ``robots/<name>/``, register its config with
``@RobotConfig.register_subclass("<name>")``, and import it below. Shared
code should not need to change — if it does, that is a sign something
robot-specific leaked into it.

WHY THE DISTRIBUTION IS NAMED ``lerobot_robot_sparklab``
========================================================
Every ``lerobot-*`` CLI calls ``register_third_party_plugins()`` at startup,
which scans installed *distributions* for names beginning with
``lerobot_robot_`` / ``lerobot_teleoperator_`` / ``lerobot_camera_`` /
``lerobot_policy_`` and imports the matching top-level module — this one.
Importing it runs the registration decorators below, so every SparkLab robot
becomes a valid ``--robot.type=...`` with no wrapper script. One distribution
covers the whole lab; the prefix is load-bearing, do not rename it away.

Note this is separate from how LeRobot *locates* a Robot class once its
config is chosen: ``make_device_from_device_class`` looks only in the
**direct parent package** of the config's module, never the package root. So
each robot subpackage re-exports its own Robot/Config pair — see
``robots/yam_ultra/__init__.py``.
"""

__version__ = "0.1.0"

# Import side effect: runs each robot's @RobotConfig.register_subclass and
# @TeleoperatorConfig.register_subclass decorators. Add new robots here.
from .robots import yam_ultra  # noqa: F401

__all__ = ["yam_ultra"]
