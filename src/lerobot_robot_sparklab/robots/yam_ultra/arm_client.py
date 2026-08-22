"""Client for one ``arm_server`` process, shaped like an i2rt robot.

One per arm, held by ``YamUltraFollower``; the only thing in the LeRobot
process that talks to the motors. Duck-types the subset of ``MotorChainRobot``
the follower calls, so moving the arms out of process left its logic untouched.

Two calling styles: blocking ``get_joint_pos()`` / ``command_joint_pos()`` for
anything off the hot path, and ``read_async()`` / ``command_clamped_async()``
returning portal futures, so a bimanual caller can start both arms before
waiting on either. Liveness rides back on every response — see DESIGN.md.
"""

from __future__ import annotations

import logging
import socket
import time

import numpy as np
import portal

logger = logging.getLogger(__name__)

_CALL_TIMEOUT_S = 30.0

class ArmServerUnreachable(RuntimeError):
    """Raised when no ``arm_server`` is listening; the message says how to start one."""


class YamArmClient:
    """One arm, over portal RPC."""

    def __init__(self, host: str, port: int, connect_timeout_s: float = 10.0,
                 expect_sim: bool | None = None,
                 expect_channel: str | None = None) -> None:
        self.addr = f"{host}:{port}"
        self._alive = True
        self._await_listener(host, port, connect_timeout_s)
        self._client = portal.Client(self.addr, name=f"yam-client-{port}")
        self._client.connect(timeout=connect_timeout_s)
        ident = self._client.call("identity", {}).result(timeout=connect_timeout_s)
        self._n_dofs = int(ident["n"])
        self.sim = bool(ident["sim"])
        self.channel = str(ident["channel"])

        if expect_sim is not None and self.sim != expect_sim:
            self.close()
            raise ArmServerUnreachable(
                f"arm_server at {self.addr} is running in "
                f"{'SIM' if self.sim else 'REAL'} mode but this follower expects "
                f"{'SIM' if expect_sim else 'REAL'}.\n"
                f"Almost always a stale server left on the port. Stop it and "
                f"restart the right one:\n"
                f"    ss -ltnp | grep {port}      # find the pid\n"
                f"    kill <pid>\n"
                f"    ./scripts/start_arm_servers.sh{'  --sim' if expect_sim else ''}"
            )
        # A warning, not an error: the follower cannot know how a rig names its
        # interfaces. But a swap means left/right are mirrored.
        if expect_channel is not None and self.channel != expect_channel:
            logger.warning(
                "arm_server at %s is driving %r, but this follower has it configured "
                "as %r. If left/right are swapped the arms will mirror each other — "
                "check which server owns which CAN channel.",
                self.addr, self.channel, expect_channel)

        logger.info("connected to arm_server at %s (%s, channel=%s, %d dofs)",
                    self.addr, "SIM" if self.sim else "REAL", self.channel, self._n_dofs)

    @staticmethod
    def _await_listener(host: str, port: int, timeout_s: float) -> None:
        """Block until something accepts TCP on host:port, or raise.

        Retries rather than probing once: the follower's sim mode spawns its
        servers moments before connecting and they take ~1 s to come up.
        """
        deadline = time.monotonic() + timeout_s
        last_err: OSError | None = None
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return
            except OSError as e:
                last_err = e
                time.sleep(0.2)
        raise ArmServerUnreachable(
            f"no arm_server listening on {host}:{port} after {timeout_s:.0f}s "
            f"({last_err}).\n"
            f"The follower connects to the arm servers, it does not start them.\n"
            f"Start them first:\n"
            f"    ./scripts/start_arm_servers.sh\n"
            f"or for one arm:\n"
            f"    python -m lerobot_robot_sparklab.robots.yam_ultra.arm_server "
            f"--channel <can_left|can_right> --port {port}"
        )

    # ---- i2rt-robot-shaped surface ------------------------------------
    def num_dofs(self) -> int:
        return self._n_dofs

    def get_joint_pos(self) -> np.ndarray:
        out = self._client.call("read", {}).result(timeout=_CALL_TIMEOUT_S)
        self._alive = bool(out["alive"])
        return np.asarray(out["pos"], dtype=float)

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        out = self._client.call(
            "command", {"pos": np.asarray(joint_pos, dtype=np.float64)}
        ).result(timeout=_CALL_TIMEOUT_S)
        self._alive = bool(out["alive"])

    # ---- combined read+clamp+command (one round trip) -----------------
    def read_async(self):
        """Issue a read and return the future, so a caller can overlap both arms."""
        return self._client.call("read", {})

    def command_clamped_async(self, target: np.ndarray, caps: np.ndarray | None,
                              n_arm: int):
        """Issue read+clamp+command as one call; returns the future.

        Halves the follower's RPC rate, and removes the gap between reading
        `present` and commanding during which the arm could move out from
        under the clamp.

        target: absolute joint positions.
        caps: per-joint Δq allowance, or None to skip clamping.
        n_arm: how many leading joints the caps apply to.
        """
        payload = {"target": np.asarray(target, dtype=np.float64),
                   "n_arm": np.int64(n_arm)}
        if caps is not None:
            payload["caps"] = np.asarray(caps, dtype=np.float64)
        return self._client.call("command_clamped", payload)

    def collect(self, future, timeout_s: float = _CALL_TIMEOUT_S) -> dict:
        """Await a future from the *_async helpers and refresh liveness."""
        out = future.result(timeout=timeout_s)
        if "alive" in out:
            self._alive = bool(out["alive"])
        return out

    def close(self) -> None:
        """Drop the RPC connection only.

        Does not stop the server: it owns torque and its own park-on-exit, and
        other clients may still want it. The follower parks before calling this.
        """
        try:
            self._client.close()
        except Exception:
            logger.exception("error closing RPC client for %s", self.addr)

    # ---- liveness -----------------------------------------------------
    def is_alive(self) -> bool:
        """Cached; refreshed by every read/command."""
        return self._alive

    def park_async(self, duration_s: float = 5.0):
        """Start the server-side ramp home and return the portal future.

        Separate from ``park`` because the ramp blocks server-side, so a
        bimanual caller must start both arms before awaiting either.
        """
        return self._client.call("park", {"duration_s": np.float64(duration_s)})

    def park(self, duration_s: float = 5.0) -> bool:
        """Ramp home and wait. Returns whether the server reported success."""
        out = self.park_async(duration_s).result(timeout=duration_s + _CALL_TIMEOUT_S)
        return bool(out["ok"])

    def __repr__(self) -> str:
        return f"YamArmClient({self.addr}, dofs={self._n_dofs}, alive={self._alive})"
