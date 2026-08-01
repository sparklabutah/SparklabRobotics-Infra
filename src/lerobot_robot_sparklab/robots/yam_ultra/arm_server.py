"""One YAM-Ultra arm, served over portal RPC from its OWN process.

HOW THE WHOLE THING FITS TOGETHER
=================================

Three processes, not one::

    ┌─ lerobot-rollout / lerobot-record process ──────────────┐
    │                                                          │
    │   policy inference (GPU, ~200ms)                         │
    │   cameras (RealSense, own bg threads)                    │
    │   YamUltraFollower        <- LeRobot Robot interface     │
    │     └── YamArmClient x2   <- arm_client.py               │
    └───────────┬──────────────────────────┬───────────────────┘
                │ portal RPC (localhost)   │
    ┌───────────▼────────────┐  ┌──────────▼─────────────┐
    │ arm_server --can_left  │  │ arm_server --can_right │
    │   ArmServer (this file)│  │   ArmServer            │
    │   └ i2rt MotorChainRobot│ │   └ i2rt MotorChainRobot│
    │     └ CAN thread ~88Hz  │ │     └ CAN thread ~88Hz  │
    └────────────────────────┘  └────────────────────────┘
              ↕ CAN                        ↕ CAN
           left arm motors            right arm motors

WHY SPLIT AT ALL. i2rt's CAN thread must keep sending inside each motor's
own watchdog window. In one process it shared a GIL with policy inference
and got starved, and the motors reported ``loss communication`` (error 0xD)
and killed the control loop mid-run — while every SocketCAN fault counter
read zero, i.e. the wire was fine and the frames just weren't being sent
often enough. Separate process = separate GIL = nothing the policy does can
starve CAN.

WHO OWNS WHAT. This process owns the motors, the CAN link, and the torque.
It holds no policy: no Δq clamp of its own, no gripper flip, no cameras, no
idea when to park. Those all stay follower-side. The one thing it decides
for itself is parking on its own shutdown (see SAFETY below), because the
follower cannot help if this process is killed directly.

WHAT ONE TICK LOOKS LIKE (2 round trips per arm, issued in parallel)::

    follower.get_observation()
        -> "read"            -> {pos[7], alive}          # joint angles
    follower.send_action(a)
        -> "command_clamped" -> {sent[7], overshoot, alive}
             server side: read present, clip step to caps, command motors

    The follower computes `caps` (it depends on the follower's measured
    tick interval, which this process cannot know) and sends them along;
    this end only applies them. Read+clamp+command happen under one lock so
    the arm cannot move between the read and the command.

THE RPC SURFACE is deliberately tiny — ``identity``/``num_dofs`` at connect,
then ``read``/``command``/``command_clamped``/``park``. Everything is plain
numpy in dicts; portal packs it.

Giving each arm its own process gives its CAN thread its own GIL, so nothing
in the LeRobot process can starve it. The architecture (and the choice of
portal, already an i2rt dependency) is borrowed from whats2000/lerobot's
``bi_yam_follower``; the safety behaviour below is ours.

KNOWN COST: portal's client socket spins rather than blocking while a call
is in flight, burning ~1 core per client (measured 93% at 4.4 Hz, 118% at
20 Hz — nearly independent of call rate, so it is idle spin, not per-call
work). Two clients means ~2 cores of GIL-holding Python competing with the
control loop. Merging both arms into ONE server would halve that and still
keep CAN out of the policy process.

Run one per arm — the follower connects to them, it does not start them:

    python -m lerobot_robot_sparklab.robots.yam_ultra.arm_server \\
        --channel can_left --port 11333

``--sim`` backs it with an i2rt SimRobot instead (no CAN, nothing physical
moves), which is what ``YamUltraFollowerConfig.sim`` auto-spawns for tests.
``scripts/start_arm_servers.sh`` launches both arms at the ports the
follower defaults to.

SAFETY: this process owns the torque. i2rt's ``close()`` cuts it wherever
the arm stands, so Ctrl-C here would drop the arm even though the follower
never asked for that. ``--park-on-exit`` (default on) ramps back to the pose
captured at startup — the folded pose the arms power up in — before closing,
so killing the server directly is as safe as a clean follower disconnect.
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
    """Wraps one i2rt robot and exposes the slice of it the follower needs.

    Deliberately a *thin* surface: every policy decision (Δq clamping,
    gripper flip, camera handling, when to park) stays in the follower, so
    this process has no opinion beyond "own the CAN loop and keep the arm
    from dropping". The one exception is ``park``, which is duplicated here
    purely as a shutdown safety net — see the module docstring.
    """

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

        # Informational only — park() targets zero, not this. Logged because
        # a startup pose far from zero means the arm was not left folded, and
        # the first park will therefore be a long move.
        q = np.asarray(self._robot.get_joint_pos(), dtype=float)
        logger.info("%s pose at startup: %s (park target is zero)",
                    channel, np.round(q[:ARM_JOINTS], 3).tolist())

    # ---- liveness -----------------------------------------------------
    def _alive(self) -> bool:
        """False once i2rt's control loop has died (motor error / loss of
        communication). SimRobots have no motor_chain — treat as alive.

        Returned with every read/command rather than exposed as its own call
        so the follower can track it per tick without paying an extra
        round trip: a dead chain keeps answering get_joint_pos() with the
        last pose it read, so nothing else would reveal it.
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

        Nothing about the RPC surface distinguishes a SimRobot from real
        motors — ``read``/``command`` look identical either way. An orphaned
        ``--sim`` server left listening on the real port would therefore
        absorb a hardware run silently: the follower would report a connected
        robot, the policy would appear to run, and the real arms would never
        move. (Observed: two orphaned sim servers held these ports across a
        real rollout.) The client checks this at connect.
        """
        return {"n": np.int64(self._n_dofs),
                "sim": np.bool_(self.sim),
                "channel": self.channel}

    def read(self, _data=None) -> dict:
        with self._lock:
            pos = np.asarray(self._robot.get_joint_pos(), dtype=np.float64)
            return {"pos": pos, "alive": np.bool_(self._alive())}

    def command(self, data: dict) -> dict:
        """Command absolute joint positions. Returns liveness so the caller
        can refresh its cached view without a second call."""
        pos = np.asarray(data["pos"], dtype=float)
        with self._lock:
            self._robot.command_joint_pos(pos)
            return {"alive": np.bool_(self._alive())}

    def command_clamped(self, data: dict) -> dict:
        """Read present pose, clamp the requested step, command — one call.

        Exists to halve the RPC rate. portal's client socket costs ~97% of a
        core at 20 calls/s and ~12% idle (measured), and that thread holds the
        GIL, so every round trip we remove buys back control-loop headroom.
        This merges the follower's read-then-command pair.

        The follower still owns the *policy*: it computes ``caps`` (which
        depend on ITS measured tick interval, something this process cannot
        know) and passes them in. This end only applies them. Doing the read
        and the command under one lock also closes the window where the arm
        could move between the two, which the split version had.

        Returns the pose actually commanded, so the caller needs no follow-up
        read, plus how far the request overshot the cap so it can warn.
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

        Zero is the arm's canonical rest configuration — i2rt treats it as
        such throughout (its joint-limit-violation message tells you to "move
        the arm to zero position and power cycle the robot"), and it is the
        pose the arms power up folded into. Targeting it rather than a pose
        captured at connect matters: a captured pose is just wherever the arm
        happened to be, so a run that ended mid-air would make "park" ramp to
        mid-air and then cut torque.

        Uses i2rt's own ``MotorChainRobot.move_joints()``, which already
        interpolates in joint space, instead of re-rolling a ramp here.

        NOTE: the target includes the gripper (element 7), so parking also
        drives the gripper to 0. If you need it to hold what it is carrying,
        target ``[zeros(ARM_JOINTS), current_gripper]`` instead.
        """
        duration_s = float(data.get("duration_s", 5.0)) if data else 5.0
        # Never let this raise out of the RPC handler: portal tears the
        # connection down on a server-side exception, and this process owns
        # the torque — losing it mid-park is exactly when the arm drops.
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
                    # MotorChainRobot: i2rt already interpolates in joint space.
                    self._robot.move_joints(target, time_interval_s=duration_s)
                else:
                    # SimRobot has no move_joints (see i2rt/robots/sim_robot.py —
                    # it implements command_joint_pos and little else), so mirror
                    # move_joints' own 50-step linear interpolation here rather
                    # than teleporting, which would make sim behave nothing like
                    # hardware.
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
