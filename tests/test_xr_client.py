import unittest

from lerobot_robot_sparklab.quest.xr_client import XRFrameClient


class XRFrameClientLifecycleTest(unittest.TestCase):
    def test_connect_timeout_stops_background_thread(self) -> None:
        client = XRFrameClient("ws://invalid", name="timeout-test")

        def wait_for_stop() -> None:
            assert client._stop is not None
            client._stop.wait()

        client._main = wait_for_stop  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "timed out connecting"):
            client.connect(timeout_s=0.01)

        self.assertIsNone(client._thread)
        self.assertFalse(client.is_connected)


if __name__ == "__main__":
    unittest.main()
