"""bi_gello records what the GELLO nodes command, and the robot then commands nothing."""

import threading
import time

import numpy as np
import pytest
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

from lerobot_robot_sparklab import ros
from lerobot_robot_sparklab.follower import YamUltraFollower, YamUltraFollowerConfig
from lerobot_robot_sparklab.teleoperator import BiGelloTeleoperator, BiGelloTeleoperatorConfig
from sparklab_yam.constants import DEFAULT_MAX_RELATIVE_TARGET, HANDS

PRESENT = np.array([0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 0.8])


@pytest.fixture
def driver():
    """Both arms' joint_states at PRESENT, as the drivers publish them."""
    ros.init()
    node = Node("fake_drivers")
    pubs = {h: node.create_publisher(JointState, f"/{h}/joint_states", 10) for h in HANDS}
    stop = threading.Event()

    def publish():
        while not stop.is_set():
            msg = JointState(position=PRESENT.tolist())
            msg.header.stamp = node.get_clock().now().to_msg()
            for pub in pubs.values():
                pub.publish(msg)
            time.sleep(0.02)
    thread = threading.Thread(target=publish)
    thread.start()
    yield node
    stop.set()
    thread.join()
    node.destroy_node()


def test_it_refuses_to_start_without_the_gello_nodes():
    teleop = BiGelloTeleoperator(BiGelloTeleoperatorConfig())
    with pytest.raises(RuntimeError, match="teleop.launch.py cameras:=true"):
        teleop.connect(timeout_s=0.5)
    assert not teleop.is_connected


def test_the_action_is_the_gello_command_clamped_as_the_driver_clamps_it(driver):
    arm = {h: driver.create_publisher(Float64MultiArray, f"/{h}/arm/command", 10) for h in HANDS}
    gripper = {h: driver.create_publisher(Float64, f"/{h}/gripper/command", 10) for h in HANDS}
    robot = YamUltraFollower(YamUltraFollowerConfig(camera_ids=[], command_arms=False))
    teleop = BiGelloTeleoperator(BiGelloTeleoperatorConfig())
    try:
        robot.connect()
        teleop.connect()
        assert teleop.get_action() == {}
        held = robot.send_action(teleop.get_action())
        assert [held[f"left_joint_{j}.pos"] for j in range(1, 7)] == PRESENT[:6].tolist()

        targets = {"left": PRESENT[:6] + 0.05, "right": PRESENT[:6] + [1.0, 0, 0, 0, 0, 0]}
        deadline = time.monotonic() + 5.0
        while len(teleop.get_action()) < 14 and time.monotonic() < deadline:
            for h in HANDS:
                arm[h].publish(Float64MultiArray(data=targets[h].tolist()))
                gripper[h].publish(Float64(data=0.25))
            time.sleep(0.05)
        sent = robot.send_action(teleop.get_action())
        np.testing.assert_allclose([sent[f"left_joint_{j}.pos"] for j in range(1, 7)], targets["left"])
        assert sent["right_joint_1.pos"] == pytest.approx(PRESENT[0] + DEFAULT_MAX_RELATIVE_TARGET[0])
        assert sent["left_gripper.pos"] == sent["right_gripper.pos"] == 0.25
        # Only the GELLO side publishes commands, as the GELLO nodes' arm check requires.
        topics = [f"/{h}/{t}/command" for h in HANDS for t in ("arm", "gripper")]
        assert [driver.count_publishers(t) for t in topics] == [1] * 4
        t0 = time.monotonic()
        robot.park()
        assert time.monotonic() - t0 < 0.1
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        robot.disconnect()
