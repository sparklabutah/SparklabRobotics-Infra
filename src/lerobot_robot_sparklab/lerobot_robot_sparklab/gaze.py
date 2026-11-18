"""The Aria wearer's gaze as Gaze4Collab's columns, each frame given what a rollout would have then.

Reads /gaze_mapping/gaze from aria_ros2_bridge's gaze_mapping node: the gaze point in left_base,
published only when its ray ends on a surface the top camera sees.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo

from . import ros

CAMS = ("left_wrist", "right_wrist", "top", "face")
"""Gaze4Collab's gaze_px order. Only top is projected here; the others stay -1."""
RADIUS = 0.04  # m the gaze may wander and still dwell, Gaze4Collab's robot.gaze.radius
# s since the newest gaze sample arrived, past which the gaze is off; a guess, not measured on the glasses.
STALE_S = 0.1
LIVE_S = 0.5

FEATURES = {
    "observation.gaze": {"dtype": "float32", "shape": (5,), "names": ["x", "y", "z", "on", "dwell"]},
    "observation.gaze_px": {"dtype": "float32", "shape": (8,),
                            "names": [f"{c}_{a}" for c in CAMS for a in "uv"]},
}


def _top_in_left_base() -> np.ndarray:
    path = Path(get_package_share_directory("sparklab_yam")) / "config" / "extrinsics.yaml"
    t = next(t for t in yaml.safe_load(path.read_text())["transforms"]
             if (t["parent"], t["child"]) == ("left_base", "top"))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(t["rotation"]).as_matrix()
    T[:3, 3] = t["translation"]
    return T


class GazeTracker:
    """The wearer's gaze frame by frame, as it arrives.

    fps: the dataset's frame rate, the dwell's step.
    """

    def __init__(self, fps: int):
        self.dt = 1.0 / fps
        top = _top_in_left_base()
        self._rvec = cv2.Rodrigues(top[:3, :3].T)[0]
        self._tvec = -top[:3, :3].T @ top[:3, 3]
        self._latest: tuple[float, np.ndarray] | None = None
        self._camera = None
        self.close = lambda: None
        self.start()

    def connect(self, timeout_s: float = 10.0) -> None:
        """Subscribe, and wait for the top camera's intrinsics."""
        ros.init()
        node = Node("gaze_tracker")
        node.create_subscription(PointStamped, "/gaze_mapping/gaze", self._on_gaze, 10)
        node.create_subscription(CameraInfo, "/cameras/top/color/camera_info", self._on_info, 1)
        self.close = ros.spin_in_background(node)
        deadline = time.monotonic() + timeout_s
        while self._camera is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._camera is None:
            self.close()
            raise RuntimeError("nothing on /cameras/top/color/camera_info: is the rig's camera node up?")

    def _on_gaze(self, msg: PointStamped) -> None:
        p = msg.point
        self._latest = (time.monotonic(), np.array([p.x, p.y, p.z]))

    def _on_info(self, msg: CameraInfo) -> None:
        self._camera = (np.array(msg.k).reshape(3, 3), np.array(msg.d), msg.width, msg.height)

    def live(self) -> bool:
        """Whether a gaze point came in the last LIVE_S."""
        return self._latest is not None and time.monotonic() - self._latest[0] < LIVE_S

    def start(self) -> None:
        """Begin an episode: no dwell."""
        self._on: list[float] = []
        self._centre, self._count, self._dwell = None, 0, 0.0

    def share_on(self) -> float:
        """The share of this episode's frames with gaze."""
        return float(np.mean(self._on)) if self._on else 0.0

    def step(self) -> dict[str, np.ndarray]:
        """One frame: the newest gaze sample, if it arrived within STALE_S, and the dwell.

        Returns: the frame's columns, as FEATURES.
        """
        latest = self._latest
        on = latest is not None and time.monotonic() - latest[0] <= STALE_S
        p = latest[1] if on else None
        if on and self._centre is not None and np.linalg.norm(p - self._centre) <= RADIUS:
            self._count += 1
            self._centre += (p - self._centre) / self._count
            self._dwell += self.dt
        elif not on and self._centre is None and self._count:
            self._dwell += self.dt
        else:
            self._centre, self._count, self._dwell = (p.copy(), 1, 0.0) if on else (None, 1, 0.0)
        gaze = np.array([0.0, 0.0, 0.0, 0.0, self._dwell], np.float32)
        px = np.full(2 * len(CAMS), -1.0, np.float32)
        if on:
            gaze[:4] = [*p, 1.0]
            px[4:6] = self._pixel(p)
        self._on.append(float(on))
        return {"observation.gaze": gaze, "observation.gaze_px": px}

    def _pixel(self, p: np.ndarray) -> np.ndarray:
        K, d, width, height = self._camera
        uv = cv2.projectPoints(p.reshape(1, 3), self._rvec, self._tvec, K, d)[0].ravel()
        return uv if 0 <= uv[0] < width and 0 <= uv[1] < height else np.array([-1.0, -1.0])
