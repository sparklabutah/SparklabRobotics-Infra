import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from lerobot_robot_sparklab.robots.yam_ultra.teleop.bi_quest_teleop import (
    BiQuestTeleoperator,
    BiQuestTeleoperatorConfig,
    _RerunPoseLogger,
    _headset_relative_calibration,
)


class QuestTeleopStateTest(unittest.TestCase):
    def make_single(self) -> BiQuestTeleoperator:
        config = BiQuestTeleoperatorConfig(
            id="test",
            publish_ik_state=False,
            rest_ramp_duration_s=1.0,
        )
        return BiQuestTeleoperator(config, hands=("right",))

    def test_single_hand_builds_only_one_controller(self) -> None:
        teleop = self.make_single()
        self.assertEqual(tuple(teleop._arms), ("right",))
        self.assertEqual(len(teleop.action_features), 7)
        self.assertTrue(all(key.startswith("right_") for key in teleop.action_features))

    def test_stow_ramp_advances_with_a_stale_pose(self) -> None:
        teleop = self.make_single()
        arm = teleop._arms["right"]
        arm["qpos"][:6] = 1.0
        teleop._stow_arm("right", arm)
        arm["ramp_start_t"] = time.perf_counter() - 0.5

        progressed = teleop._advance_ramp(
            "right",
            arm,
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            pose_is_fresh=False,
        )

        self.assertTrue(progressed)
        self.assertTrue(arm["ramp_active"])
        np.testing.assert_allclose(arm["qpos"][:6], 0.5 * (1.0 + arm["ramp_target_q"]), atol=0.03)

    def test_stow_completion_does_not_reanchor_from_stale_pose(self) -> None:
        teleop = self.make_single()
        arm = teleop._arms["right"]
        arm["engaged"] = True
        teleop._stow_arm("right", arm)
        arm["ramp_start_t"] = time.perf_counter() - 2.0

        teleop._advance_ramp(
            "right",
            arm,
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            pose_is_fresh=False,
        )

        self.assertFalse(arm["ramp_active"])
        self.assertTrue(arm["needs_reanchor"])

    def test_pose_debug_runs_without_an_armed_state(self) -> None:
        teleop = self.make_single()
        pose_debug = Mock()
        teleop._pose_debug = pose_debug
        teleop._latest_xr_frame = {
            "viewer": {"position": [0, 1.7, 0], "orientation": [0, 0, 0, 1]},
            "controllers": {
                "right": {
                    "position": [0.2, 1.2, -0.4],
                    "orientation": [0, 0, 0, 1],
                    "buttons": [],
                }
            },
        }
        teleop._last_xr_frame_time = time.time()

        teleop.get_action()

        pose_debug.log_frame.assert_called_once_with(
            teleop._latest_xr_frame, teleop._arms
        )

    def test_raiden_bimanual_transform_is_inverted_for_left_base_world(self) -> None:
        stored = np.eye(4)
        stored[:3, 3] = [0.04, 0.56, 0.01]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_results.json"
            path.write_text(
                json.dumps(
                    {
                        "bimanual_transform": {
                            "right_base_to_left_base": stored.tolist()
                        }
                    }
                )
            )

            right_base_in_left = _RerunPoseLogger._load_right_base_transform(
                str(path)
            )

        np.testing.assert_allclose(right_base_in_left, np.linalg.inv(stored))

    def test_headset_yaw_maps_head_forward_to_robot_forward(self) -> None:
        yaw = np.pi / 2
        viewer = {
            "orientation": [0.0, np.sin(yaw / 2), 0.0, np.cos(yaw / 2)]
        }
        base_r_calib = np.array(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )

        calibrated = _headset_relative_calibration(base_r_calib, viewer)

        head_forward_in_quest = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            calibrated @ head_forward_in_quest,
            [1.0, 0.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(calibrated @ calibrated.T, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
