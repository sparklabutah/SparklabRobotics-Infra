"""Robot-agnostic Meta Quest input: 6-DoF controller poses and buttons.

    client = XRFrameClient("wss://127.0.0.1:8443/ws")
    client.connect()
    frame, age_s = client.latest()
    ctrl = (frame.get("controllers") or {}).get("right")
    if buttons.pressed(ctrl["buttons"], buttons.GRIP):
        ...

Pair with ``core.pose_mapping.ClutchPoseMapper`` for clutch-relative motion.
"""

from . import buttons  # noqa: F401
from .xr_client import XRFrameClient  # noqa: F401

__all__ = ["XRFrameClient", "buttons"]
