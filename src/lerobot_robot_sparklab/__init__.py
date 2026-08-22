"""SparkLab robotics stack — shared infrastructure plus per-robot packages.

    relay/     WebSocket relay, WebRTC camera streams, the WebXR page
    quest/     Robot-agnostic Quest input
    core/      Clutch-relative pose mapping with absorbing reach limits
    cameras/   RealSense calibration, presets, intrinsics export
    tools/     No-hardware dry-run/inspection utilities
    robots/    One subpackage per robot

Adding a robot: create ``robots/<name>/``, register its config with
``@RobotConfig.register_subclass("<name>")``, and import it below.

The ``lerobot_robot_`` distribution prefix is load-bearing for plugin
discovery — see DESIGN.md before renaming it.
"""

__version__ = "0.1.0"

# Import side effect: runs each robot's registration decorators. Add robots here.
from .robots import yam_ultra  # noqa: F401

__all__ = ["yam_ultra"]
