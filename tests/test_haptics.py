import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lerobot_robot_sparklab.robots.yam_ultra.teleop.haptics import ForceHaptics


def config() -> SimpleNamespace:
    return SimpleNamespace(
        force_haptic_max_nm=1.0,
        force_haptic_velocity_comp_nm=0.0,
        force_haptic_threshold_nm_left=0.25,
        force_haptic_threshold_nm_right=0.25,
        force_haptic_enabled=True,
    )


class ForceHapticsTest(unittest.TestCase):
    def test_torque_maps_above_deadband(self) -> None:
        feedback = ForceHaptics(config(), ("left",))
        intensities, result = feedback.update({"left_gripper.torque": 0.625})
        self.assertAlmostEqual(intensities["left"], 0.5)
        self.assertIsNone(result)

    def test_calibration_updates_threshold(self) -> None:
        cfg = config()
        feedback = ForceHaptics(cfg, ("left", "right"))
        with patch("time.time", side_effect=[10.0, 11.0]):
            self.assertTrue(feedback.start_calibration(0.5))
            _, result = feedback.update({
                "left_gripper.torque": 0.1,
                "right_gripper.torque": 0.2,
            })
        self.assertIsNotNone(result)
        self.assertAlmostEqual(cfg.force_haptic_threshold_nm_left, 0.3)
        self.assertAlmostEqual(cfg.force_haptic_threshold_nm_right, 0.4)


if __name__ == "__main__":
    unittest.main()
