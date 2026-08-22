"""One YAM-Ultra arm, served over portal RPC from its OWN process.

    ┌─ lerobot-rollout / lerobot-record ──────────────────────┐
    │  policy inference (GPU) · cameras · YamUltraFollower    │
    │    └── YamArmClient x2                                  │
    └───────────┬──────────────────────────┬──────────────────┘
                │ portal RPC (localhost)   │
    ┌───────────▼────────────┐  ┌──────────▼─────────────┐
    │ arm_server can_left    │  │ arm_server can_right   │
    │   └ i2rt MotorChainRobot, CAN thread ~88 Hz        │
    └────────────────────────┘  └────────────────────────┘

This process owns the motors, the CAN link and the torque, and holds no policy:
the Δq clamp, gripper flip, cameras and park timing all stay follower-side. The
exception is parking on its own shutdown, which the follower cannot do if this
process is killed directly. See DESIGN.md for why the split exists.

One tick is two round trips per arm, issued in parallel: ``read`` for joint
angles, ``command_clamped`` for read+clamp+command under one lock.

Run one per arm; the follower connects to them, it does not start them:

    python -m lerobot_robot_sparklab.robots.yam_ultra.arm_server \\
        --channel can_left --port 11333

``--sim`` backs it with an i2rt SimRobot. ``scripts/start_arm_servers.sh``
launches both at the ports the follower defaults to.

SAFETY: i2rt's ``close()`` cuts torque wherever the arm stands, so Ctrl-C would
drop it. ``--park-on-exit`` (default on) ramps to the startup pose first.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time

import numpy as np
import portal

from .connection import connect_yam_arm

logger = logging.getLogger(__name__)

ARM_JOINTS = 6


class ArmServer:
    """Wraps one i2rt robot and exposes the slice of it the follower needs."""

    def __init__(self, channel: str, sim: bool = False,
                 enable_auto_recovery: bool = False) -> None:
        self.channel = channel
        self.sim = sim
        logger.info("connecting %s%s (gripper self-homes, keep clear)",
                    channel, " (SIM)" if sim else "")
        self._robot = connect_yam_arm(channel, sim=sim,
                                      enable_auto_recovery=enable_auto_recovery)
        self._n_dofs = int(self._robot.num_dofs())
        self._lock = threading.Lock()

        # A startup pose far from zero means the arm was not left folded, so
        # the first park will be a long move.
        q = np.asarray(self._robot.get_joint_pos(), dtype=float)
        logger.info("%s pose at startup: %s (park target is zero)",
                    channel, np.round(q[:ARM_JOINTS], 3).tolist())

    # ---- liveness -----------------------------------------------------
    def _alive(self) -> bool:
        """False once i2rt's control loop has died; SimRobots count as alive.

        Rides along with every read/command rather than being its own call,
        since nothing else reveals a dead chain.
        """
        chain = getattr(self._robot, "motor_chain", None)
        if chain is not None and not getattr(chain, "running", True):
            return False
        return True

    # ---- RPC surface --------------------------------------------------
    def num_dofs(self, _data=None) -> dict:
        return {"n": np.int64(self._n_dofs)}

    def identity(self, _data=None) -> dict:
        """What this server actually is, so a client can refuse a mismatch.

        Nothing else distinguishes a SimRobot from real motors, so an orphaned
        ``--sim`` server on the real port would absorb a hardware run silently.
        """
        return {"n": np.int64(self._n_dofs),
                "sim": np.bool_(self.sim),
                "channel": self.channel}

    def read(self, _data=None) -> dict:
        with self._lock:
            pos = np.asarray(self._robot.get_joint_pos(), dtype=np.float64)
            return {"pos": pos, "alive": np.bool_(self._alive())}

    def command(self, data: dict) -> dict:
        """Command absolute joint positions. Returns liveness, so the caller
        can refresh its cached view without a second call."""
        pos = np.asarray(data["pos"], dtype=float)
        with self._lock:
            self._robot.command_joint_pos(pos)
            return {"alive": np.bool_(self._alive())}

    def command_clamped(self, data: dict) -> dict:
        """Read present pose, clamp the requested step, command — one call.

        Halves the RPC rate and closes the window where the arm could move
        between the read and the command.

        data: `target` absolute positions, optional `caps`, optional `n_arm`.
        Returns: `sent` (pose commanded), `overshoot` (how far the request
            exceeded the cap, per joint) and `alive`.
        """
        target = np.asarray(data["target"], dtype=float)
        caps = data.get("caps")
        n_arm = int(data.get("n_arm", ARM_JOINTS))
        with self._lock:
            present = np.asarray(self._robot.get_joint_pos(), dtype=float)
            cmd = target.copy()
            overshoot = np.zeros(n_arm)
            if caps is not None:
                caps = np.asarray(caps, dtype=float)
                step = cmd[:n_arm] - present[:n_arm]
                overshoot = np.maximum(np.abs(step) - caps, 0.0)
                cmd[:n_arm] = present[:n_arm] + np.clip(step, -caps, caps)
            self._robot.command_joint_pos(cmd)
            return {"sent": cmd.astype(np.float64),
                    "overshoot": overshoot.astype(np.float64),
                    "alive": np.bool_(self._alive())}

    def park(self, data: dict | None = None) -> dict:
        """Ramp to the zero pose and hold.

        Zero rather than a pose captured at connect, which would park a run
        that ended mid-air to mid-air. The target includes the gripper, so
        parking drives it to 0.

        data: optional `duration_s` ramp length.
        Returns: `ok`, False if the control loop is dead or the ramp failed.
        """
        duration_s = float(data.get("duration_s", 5.0)) if data else 5.0
        # Never raise out of the RPC handler: portal tears the connection down
        # on a server-side exception, and losing it mid-park drops the arm.
        try:
            with self._lock:
                if not self._alive():
                    logger.warning("park: control loop is dead — cannot park, the "
                                   "arm will drop when torque is cut. Support it by hand.")
                    return {"ok": np.bool_(False)}
                logger.info("parking %s: ramping to zero over %.1fs ...",
                            self.channel, duration_s)
                target = np.zeros(self._n_dofs)
                if hasattr(self._robot, "move_joints"):
                    self._robot.move_joints(target, time_interval_s=duration_s)
                else:
                    steps = 50
                    start = np.asarray(self._robot.get_joint_pos(), dtype=float)
                    for i in range(steps + 1):
                        a = i / steps
                        self._robot.command_joint_pos((1.0 - a) * start + a * target)
                        time.sleep(duration_s / steps)
                logger.info("parked %s at zero pose.", self.channel)
                return {"ok": np.bool_(True)}
        except Exception:
            logger.exception("park failed on %s — the arm may drop when torque is cut",
                             self.channel)
            return {"ok": np.bool_(False)}

    def close(self, park_on_exit: bool = True, duration_s: float = 5.0) -> None:
        if park_on_exit:
            try:
                self.park({"duration_s": duration_s})
            except Exception:
                logger.exception("park before close failed — the arm may drop")
        try:
            self._robot.close()
        except Exception:
            logger.exception("error closing the i2rt robot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--channel", required=True, help="CAN interface, e.g. can_left")
    ap.add_argument("--port", type=int, required=True, help="portal RPC port")
    ap.add_argument("--sim", action="store_true",
                    help="i2rt SimRobot instead of CAN hardware (nothing physical moves)")
    ap.add_argument("--enable-auto-recovery", action="store_true",
                    help="let i2rt clear + re-enable an errored motor from inside its "
                         "control loop. NOTE: recovery re-asserts the pre-error target "
                         "at full gains from inside the driver, which the follower's Δq "
                         "clamp cannot see or bound — it trades a hard stop for a snap.")
    ap.add_argument("--no-park-on-exit", action="store_true",
                    help="torque off where the arm stands instead of ramping home "
                         "(support the arm yourself)")
    ap.add_argument("--park-duration-s", type=float, default=5.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    arm = ArmServer(args.channel, sim=args.sim,
                    enable_auto_recovery=args.enable_auto_recovery)

    server = portal.Server(args.port, name=f"yam-{args.channel}")
    server.bind("num_dofs", arm.num_dofs)
    server.bind("identity", arm.identity)
    server.bind("read", arm.read)
    server.bind("command", arm.command)
    server.bind("command_clamped", arm.command_clamped)
    server.bind("park", arm.park)

    stop = threading.Event()

    def _shutdown(signum, _frame):
        logger.info("signal %s — shutting down %s", signum, args.channel)
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.start(block=False)
    logger.info("%s serving on port %d — Ctrl-C to stop%s",
                args.channel, args.port,
                "" if args.no_park_on_exit else " (will park home first)")
    try:
        stop.wait()
    finally:
        try:
            server.close()
        except Exception:
            logger.exception("error closing the RPC server")
        arm.close(park_on_exit=not args.no_park_on_exit,
                  duration_s=args.park_duration_s)
        logger.info("%s stopped.", args.channel)


if __name__ == "__main__":
    main()
