"""The gaze columns: the newest sample while fresh, dwell, the top pixel, and the gaze_mapping topic."""

import threading
import time

import numpy as np
import pytest
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

from lerobot_robot_sparklab import gaze, ros

K = np.array([[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def tracker():
    o = gaze.GazeTracker(fps=30)
    o._camera = (K, np.zeros(5), 1280, 720)
    return o


def ahead_of_top(metres: float, right: float = 0.0) -> np.ndarray:
    """The point on the top camera's axis, moved right in its image, in left_base."""
    return (gaze._top_in_left_base() @ [right, 0.0, metres, 1.0])[:3]


def look(o: gaze.GazeTracker, p: np.ndarray | None, age: float = 0.0) -> dict:
    o._latest = None if p is None else (time.monotonic() - age, p)
    return o.step()


def test_a_frame_takes_the_newest_sample_while_it_is_fresh(tracker):
    cols = look(tracker, ahead_of_top(0.8))
    np.testing.assert_allclose(cols["observation.gaze"], [*ahead_of_top(0.8), 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(cols["observation.gaze_px"], [-1, -1, -1, -1, 640, 360, -1, -1], atol=1e-3)
    assert (look(tracker, ahead_of_top(0.8, 5.0))["observation.gaze_px"] == -1).all()
    cols = look(tracker, ahead_of_top(0.8), age=gaze.STALE_S + 0.05)
    assert cols["observation.gaze"][3] == 0.0 and (cols["observation.gaze_px"] == -1).all()
    assert tracker.share_on() == 2 / 3


def test_dwell_grows_on_a_fixation_and_restarts_on_a_jump_or_when_the_gaze_goes(tracker):
    dwell = [look(tracker, ahead_of_top(0.8, 0.01 * (i % 2)))["observation.gaze"][4] for i in range(10)]
    dwell += [look(tracker, ahead_of_top(0.8, 0.1))["observation.gaze"][4] for _ in range(3)]
    dwell += [look(tracker, None)["observation.gaze"][4] for _ in range(4)]
    np.testing.assert_allclose(dwell, np.r_[0:10, 0:3, 0:4] * tracker.dt, atol=1e-6)


def test_it_hears_the_gaze_mapping_node():
    ros.init()
    node = Node("fake_gaze_mapping")
    info = node.create_publisher(CameraInfo, "/cameras/top/color/camera_info", 1)
    point = node.create_publisher(PointStamped, "/gaze_mapping/gaze", 10)
    stop = threading.Event()

    def publish():
        while not stop.is_set():
            info.publish(CameraInfo(width=1280, height=720, k=K.ravel().tolist(), d=[0.0] * 5))
            msg = PointStamped()
            msg.header.stamp, msg.header.frame_id = node.get_clock().now().to_msg(), "left_base"
            msg.point.x, msg.point.y, msg.point.z = ahead_of_top(0.8).tolist()
            point.publish(msg)
            time.sleep(1 / 30)
    thread = threading.Thread(target=publish)
    thread.start()
    tracker = gaze.GazeTracker(fps=30)
    try:
        tracker.connect()
        deadline = time.monotonic() + 5.0
        while not tracker.live() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert tracker.live()
        np.testing.assert_allclose(tracker.step()["observation.gaze_px"][4:6], [640, 360], atol=1e-3)
    finally:
        stop.set()
        thread.join()
        tracker.close()
        node.destroy_node()
