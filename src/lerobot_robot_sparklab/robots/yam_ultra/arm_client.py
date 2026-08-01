"""Client for one ``arm_server`` process, shaped like an i2rt robot.

WHERE THIS SITS. One of these per arm, held by ``YamUltraFollower``. It is
the *only* thing in the LeRobot process that talks to the motors — see
``arm_server`` for the three-process picture and why the split exists.

THE TRICK. This class pretends to BE an i2rt robot. It exposes the same four
methods the follower used to call directly on ``MotorChainRobot``::

    num_dofs()  get_joint_pos()  command_joint_pos()  close()

so moving the arms into another process required almost no change to the
follower. All the hard-won follower logic — Δq clamp, park-on-disconnect,
dead-loop detection, gripper flip, camera handling — still runs where it
always did. Only the transport underneath changed.

TWO CALLING STYLES:

  blocking      ``get_joint_pos()`` / ``command_joint_pos()`` — simple,
                one round trip, used by park and by anything not on the
                hot path.

  async/paired  ``read_async()`` / ``command_clamped_async()`` return portal
                *futures*; you collect them later with ``collect()``. The
                follower uses these so it can start BOTH arms' calls before
                waiting on either — otherwise left and right pay their round
                trips back to back and the tick costs twice as much.

  ``command_clamped_async`` also folds read+clamp+command into ONE call
  instead of read-then-command, which halves round trips and removes the
  window where the arm could move between the two.

Deliberately duck-types the subset of ``i2rt.robots.MotorChainRobot`` that
``YamUltraFollower`` actually calls — ``num_dofs()`` / ``get_joint_pos()`` /
``command_joint_pos()`` / ``close()`` — so moving the arms out of process
did not require rewriting the follower's logic. Everything hard-won there
(the Δq clamp, park-on-disconnect, dead-loop detection, gripper flip,
camera handling) is untouched and still runs follower-side; only where the
CAN loop *lives* changed. See ``arm_server`` for why that move was needed.

Liveness is the one addition. i2rt exposes it as ``robot.motor_chain.running``,
an attribute the follower reads directly; over RPC that would be a round
trip on every ``is_connected`` poll, so instead every ``read``/``command``
response carries the flag and ``is_alive()`` returns the cached value. It is
refreshed on every tick either way, and a dead chain is exactly the case
where you cannot trust a separate query — ``get_joint_pos()`` keeps happily
returning the last pose it read.
"""

from __future__ import annotations

import logging
import socket
import time

import numpy as np
import portal

logger = logging.getLogger(__name__)

# Guards a connected-but-slow server (a park ramp holds the server's lock for
# park_duration_s). It does NOT guard a server that has died mid-run: portal's
# call() blocks before it hands back a future, so no result timeout can fire.
# That case is caught at connect by _await_listener; if a server dies during a
# run the follower will block rather than raise. The arm itself is safe there
# (the process owning torque is gone, so the motors' own watchdog stops them)
# — it is a liveness annoyance, not a hazard.
_CALL_TIMEOUT_S = 30.0


class ArmServerUnreachable(RuntimeError):
    """Raised when no ``arm_server`` is listening — almost always "the
    servers were not started", so the message says how to start them."""


class YamArmClient:
    """One arm, over portal RPC. See the module docstring."""

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
        # Channel is a warning, not an error: the follower cannot know how a
        # given rig names its interfaces. But a swap here means left/right are
        # mirrored, which is worth shouting about.
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

        Retries rather than probing once, because the follower's `sim` mode
        spawns its servers moments before connecting and they take ~1 s to
        come up.
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
        """Issue a read; returns the future. Lets a bimanual caller overlap
        both arms instead of paying two round trips back to back."""
        return self._client.call("read", {})

    def command_clamped_async(self, target: np.ndarray, caps: np.ndarray | None,
                              n_arm: int):
        """Issue read+clamp+command as ONE call; returns the future.

        Halves the follower's RPC rate, which matters more than it sounds:
        portal's client socket burns ~97% of a core at 20 calls/s and ~12%
        idle (measured), on a thread that holds the GIL — so RPCs removed are
        control-loop headroom returned. Also removes the gap between reading
        `present` and commanding, during which the arm used to be free to
        move out from under the clamp.
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

        Does NOT stop the server: it is a separate, externally managed
        process that owns torque and its own park-on-exit, and other clients
        (or a later run) may still want it. The follower parks through this
        client *before* calling close, exactly as it did in-process.
        """
        try:
            self._client.close()
        except Exception:
            logger.exception("error closing RPC client for %s", self.addr)

    # ---- liveness -----------------------------------------------------
    def is_alive(self) -> bool:
        """Cached — refreshed by every read/command. See module docstring."""
        return self._alive

    def park_async(self, duration_s: float = 5.0):
        """Start the server-side ramp home; returns the portal future.

        Separate from ``park`` so a bimanual caller can start BOTH arms
        before waiting on either — the ramp blocks for ``duration_s``
        server-side, so waiting on one before starting the other would move
        the arms one at a time instead of together.
        """
        return self._client.call("park", {"duration_s": np.float64(duration_s)})

    def park(self, duration_s: float = 5.0) -> bool:
        """Ramp home and wait. See ``park_async`` for the bimanual case."""
        out = self.park_async(duration_s).result(timeout=duration_s + _CALL_TIMEOUT_S)
        return bool(out["ok"])

    def __repr__(self) -> str:
        return f"YamArmClient({self.addr}, dofs={self._n_dofs}, alive={self._alive})"
