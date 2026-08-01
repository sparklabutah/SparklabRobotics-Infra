"""Robot-agnostic Meta Quest input.

The reusable half of VR teleoperation: read 6-DoF controller poses and
buttons off the relay's WebSocket stream. Nothing here knows about joints,
IK, grippers or action keys — that belongs to each robot's own teleoperator
under ``robots/<name>/teleop/``.

    from lerobot_robot_sparklab.quest import XRFrameClient, buttons

    client = XRFrameClient("wss://127.0.0.1:8443/ws")
    client.connect()
    frame, age_s = client.latest()
    ctrl = (frame.get("controllers") or {}).get("right")
    if buttons.pressed(ctrl["buttons"], buttons.GRIP):
        ...

Pair it with ``core.pose_mapping.ClutchPoseMapper`` (also shared) to turn
those poses into clutch-relative motion, then map that onto whatever your
robot's action space happens to be.
"""

from . import buttons  # noqa: F401
from .xr_client import XRFrameClient  # noqa: F401

__all__ = ["XRFrameClient", "buttons"]
