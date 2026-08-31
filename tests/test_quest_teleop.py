import time
import unittest

import numpy as np

from lerobot_robot_sparklab.robots.yam_ultra.teleop.bi_quest_teleop import (
    BiQuestTeleoperator,
    BiQuestTeleoperatorConfig,
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


if __name__ == "__main__":
    unittest.main()
