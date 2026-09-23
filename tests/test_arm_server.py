import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lerobot_robot_sparklab.robots.yam_ultra.arm_server import ArmServer


class ArmServerViewerTest(unittest.TestCase):
    def test_viewer_syncs_sim_state_and_stops_when_closed(self) -> None:
        arm = ArmServer.__new__(ArmServer)
        arm.channel = "can_test"
        arm.sim = True
        arm._robot = SimpleNamespace(
            _model=object(),
            _data=object(),
            _lock=threading.Lock(),
        )
        stop = threading.Event()
        viewer = Mock()
        viewer.is_running.side_effect = [True, False]

        with patch("mujoco.viewer.launch_passive") as launch:
            launch.return_value.__enter__.return_value = viewer
            arm.run_viewer(stop)

        launch.assert_called_once_with(arm._robot._model, arm._robot._data)
        viewer.sync.assert_called_once_with()
        self.assertTrue(stop.is_set())


if __name__ == "__main__":
    unittest.main()
