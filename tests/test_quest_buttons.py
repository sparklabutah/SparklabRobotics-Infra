import unittest

from lerobot_robot_sparklab.quest import buttons


class ButtonMappingTest(unittest.TestCase):
    def test_short_arrays_are_safe(self) -> None:
        self.assertFalse(buttons.pressed([], buttons.GRIP))
        self.assertEqual(buttons.value(None, buttons.TRIGGER), 0.0)

    def test_pressed_and_value_use_xr_standard_fields(self) -> None:
        values = [{"p": True, "v": 0.75}]
        self.assertTrue(buttons.pressed(values, buttons.TRIGGER))
        self.assertEqual(buttons.value(values, buttons.TRIGGER), 0.75)


if __name__ == "__main__":
    unittest.main()
