"""``lerobot-rollout`` with a live control channel.

Same CLI as ``lerobot-rollout`` -- every flag is forwarded to LeRobot's own
parser untouched -- plus a small HTTP control port. While the loop runs you
can retarget the policy, halt it, and send the arms home, without restarting
the process. Restarting costs ~90 s of model load, which is the whole reason
this exists.

    python -m lerobot_robot_sparklab.rollout.live --control-port 8090 \\
        --strategy.type=base --policy.path=... --robot.type=yam_ultra_sim \\
        --task="put the marker into the cardboard box" --fps 20

Drive it from another terminal with the REPL client:

    python -m lerobot_robot_sparklab.rollout.ctl

or by hand:

    curl -s localhost:8090/status
    curl -s localhost:8090/cmd -d '{"cmd":"task","arg":"pick up the scissors"}'
    curl -s localhost:8090/cmd -d '{"cmd":"reset"}'

TWO THINGS THAT ARE NOT OBVIOUS

*Changing the task must also reset the engine.* The policy emits a 30-step
chunk and the loop drains it one action at a time. Swapping ``engine._task``
alone leaves up to 30 already-queued actions -- 1.5 s at 20 Hz, longer with
interpolation -- still executing under the OLD prompt, so the arm appears to
ignore the new command and then lurch. ``retarget()`` clears the queue.

*Reset ramps, it does not teleport.* Home is reached by commanding an eased
trajectory over ``home_duration_s`` (5 s), one waypoint per tick through
``send_action``, so it still passes the follower's ``max_relative_target``
clamp like any policy action -- the clamp is a backstop, not the thing setting
the speed. A direct pose write would be fine in sim and violent on hardware;
this file is meant to drive both.

*Reset parks and HOLDS.* Homing while the loop keeps stepping means the next
tick drives the policy on the unchanged task and the arm climbs straight back
out. ``reset``/``park``/``home`` set ``held``; a new task clears it.

THREADING: HTTP handlers never touch the robot or the policy. They append to a
queue that the control loop drains between ticks -- the same discipline
``policy_server`` uses, and for the same reason: neither the robot transport
nor the inference engine is thread-safe, and a blocking call made from an HTTP
thread will deadlock the loop rather than fail.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

DEFAULT_CONTROL_PORT = 8090

# Reset target. Zeros is the YAM's straight-up rest pose and matches what the
# smoke tests use; override with --home if your bench differs.
DEFAULT_HOME = 0.0


# ---------------------------------------------------------------------------
# control state
# ---------------------------------------------------------------------------
@dataclass
class Control:
    """Shared state between the HTTP threads and the control loop.

    Commands are queued rather than applied in place so that every mutation
    happens at a known point in the tick, between reading an observation and
    sending an action. Applying a task change halfway through
    ``send_next_action`` would mix two prompts inside one chunk.
    """

    task: str = ""
    paused: bool = False
    # Parked and waiting for an instruction. Set by reset/park/home, cleared by
    # a new task (or an explicit resume).
    #
    # Distinct from `paused` because the reason differs and the operator needs
    # to see which: `paused` is "stopped mid-task, the task still stands",
    # `held` is "at home, there is nothing to do until you say so". Without it,
    # homing the arm and then leaving the loop running means the very next tick
    # steps the policy on the SAME task and the arm climbs straight back out of
    # the pose you just put it in.
    held: bool = False
    home_pos: float = DEFAULT_HOME
    reset_tolerance: float = 0.05      # rad, per joint
    # How long the ramp to home takes. Set to 0 to fall back to sending the
    # final target and letting the follower's Δq clamp rate-limit it — faster,
    # and abrupt enough to read as a jolt, which is why it is not the default.
    home_duration_s: float = 5.0
    # Budget for the arm to *settle* onto the target after the ramp, not for
    # the ramp itself.
    reset_timeout_s: float = 8.0

    _pending: queue.Queue = field(default_factory=queue.Queue)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Latest observation, for anything WATCHING rather than driving -- the web
    # harness, and any high-level agent reasoning over the scene.
    #
    # Latest-only, never a queue: a watcher that falls behind must skip frames
    # rather than build a backlog and hand an agent a scene that no longer
    # exists. Encoded in the loop thread rather than per request, so N watchers
    # cannot multiply encode cost onto the control loop.
    _frames: dict = field(default_factory=dict)      # name -> JPEG bytes
    _state: dict = field(default_factory=dict)       # joint -> radians
    _frame_t: float = 0.0
    _frame_seq: int = 0

    # counters, for /status
    ticks: int = 0
    resets: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    last_event: str = "started"

    @property
    def stepping(self) -> bool:
        """Whether the loop should be driving the policy this tick."""
        return not (self.paused or self.held)

    def submit(self, cmd: str, arg: str | None = None) -> dict:
        """Queue a command from an HTTP thread. Returns immediately."""
        cmd = (cmd or "").strip().lower()
        if cmd not in _COMMANDS:
            return {"ok": False, "error": f"unknown command {cmd!r}; "
                                          f"try {sorted(_COMMANDS)}"}
        if cmd == "task" and not (arg or "").strip():
            return {"ok": False, "error": "task needs a non-empty argument"}
        self._pending.put((cmd, arg))
        return {"ok": True, "queued": cmd}

    def drain(self) -> list[tuple[str, str | None]]:
        out = []
        while True:
            try:
                out.append(self._pending.get_nowait())
            except queue.Empty:
                return out

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "task": self.task,
                "paused": self.paused,
                "held": self.held,
                "ticks": self.ticks,
                "resets": self.resets,
                "uptime_s": round(time.perf_counter() - self.started_at, 1),
                "last_event": self.last_event,
            }

    def note(self, event: str) -> None:
        with self._lock:
            self.last_event = event

    def publish(self, obs: dict, every: int = 1) -> None:
        """Snapshot the tick's observation for watchers. Called from the loop.

        ``every`` subsamples: an agent reasoning at ~1 Hz does not need a JPEG
        encode at every 30 Hz tick, and the encode is on the control loop's
        critical path. Failures are swallowed -- a watcher missing a frame must
        never be able to interrupt the robot.
        """
        self._frame_seq += 1
        if every > 1 and self._frame_seq % every:
            return
        try:
            import cv2
            import numpy as np

            frames, state = {}, {}
            for k, v in obs.items():
                if hasattr(v, "shape") and getattr(v, "ndim", 0) == 3:
                    # LeRobot hands out RGB; cv2 encodes BGR.
                    ok, buf = cv2.imencode(".jpg", np.asarray(v)[:, :, ::-1],
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        frames[k] = buf.tobytes()
                elif isinstance(v, (int, float)):
                    state[k] = round(float(v), 5)
            with self._lock:
                self._frames = frames
                self._state = state
                self._frame_t = time.time()
        except Exception:                       # never let a watcher stall the loop
            logger.debug("publish() failed", exc_info=True)

    def observation(self) -> dict:
        """Metadata about the latest frame set (not the pixels themselves)."""
        with self._lock:
            return {
                "cameras": sorted(self._frames),
                "state": dict(self._state),
                "captured_at": self._frame_t,
                "age_s": round(time.time() - self._frame_t, 3) if self._frame_t else None,
            }

    def frame(self, name: str) -> bytes | None:
        with self._lock:
            return self._frames.get(name)


_COMMANDS = {"task", "reset", "park", "pause", "resume", "home", "status", "quit"}


# ---------------------------------------------------------------------------
# control server
# ---------------------------------------------------------------------------
def _serve(control: Control, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, payload: dict, code: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reply_jpeg(self, blob: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(blob)))
            # A watcher polling this must never see a cached frame.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self):
            path = self.path.rstrip("/")
            if path in ("/status", ""):
                self._reply(control.snapshot())
            elif path == "/observation":
                self._reply(control.observation())
            elif path.startswith("/frame/"):
                name = path[len("/frame/"):].removesuffix(".jpg")
                blob = control.frame(name)
                if blob is None:
                    self._reply({"error": f"no frame {name!r}",
                                 "cameras": control.observation()["cameras"]}, 404)
                else:
                    self._reply_jpeg(blob)
            else:
                self._reply({"error": "GET /status | /observation | /frame/<cam>.jpg"}, 404)

        def do_POST(self):
            if self.path.rstrip("/") != "/cmd":
                self._reply({"error": "POST /cmd"}, 404)
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                self._reply({"ok": False, "error": f"bad JSON: {e}"}, 400)
                return
            if payload.get("cmd") == "status":
                self._reply(control.snapshot())
                return
            self._reply(control.submit(payload.get("cmd", ""), payload.get("arg")))

        def log_message(self, *_args):
            pass          # the rollout log is noisy enough already

    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True,
                     name="rollout-control").start()
    logger.info("control channel on http://%s:%d  (POST /cmd, GET /status)",
                host, port)
    return srv


# ---------------------------------------------------------------------------
# strategy
# ---------------------------------------------------------------------------
def build_strategy_class():
    """Import LeRobot lazily and return the interactive strategy class.

    Deferred because importing ``lerobot.rollout`` pulls in torch and the
    plugin registry, which must not happen at module import time -- argv is
    rewritten first (see ``main``).
    """
    from lerobot.rollout.strategies.base import BaseStrategy
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.utils.robot_utils import precise_sleep

    class InteractiveStrategy(BaseStrategy):
        """``BaseStrategy`` whose loop checks a control queue every tick.

        The loop is reimplemented rather than wrapped because the engine's
        ``pause``/``resume`` are no-op hooks on the ABC -- ``SyncInferenceEngine``
        does not override them, so pausing has to happen in the caller. Everything
        else (observation processing, warm-up handling, action dispatch,
        telemetry) is reused from the parent unchanged.
        """

        # Subsample frame publishing: encoding sits on the control loop's
        # critical path, and an agent reasoning at ~1 Hz does not need one per
        # 30 Hz tick. Overridden from --publish-every.
        publish_every = 1

        def __init__(self, config, control: Control):
            super().__init__(config)
            self.control = control

        # -- helpers ----------------------------------------------------
        def retarget(self, task: str) -> None:
            """Point the policy at a new instruction, starting from a clean chunk.

            Also releases a hold: submitting a task is the operator saying go,
            and it is the only thing that lifts the park that reset/park/home
            leave behind.
            """
            engine = self._engine
            engine._task = task
            # Without this the queued chunk from the previous prompt keeps
            # playing; see the module docstring.
            engine.reset()
            if self._interpolator is not None:
                self._interpolator.reset()
            self._cached_obs_processed = None
            with self.control._lock:
                self.control.task = task
                was_held = self.control.held
                self.control.held = False
            logger.info("task -> %r%s", task, " (released hold)" if was_held else "")

        def hold(self, event: str) -> None:
            """Park the loop: stop stepping the policy until a task arrives."""
            with self.control._lock:
                self.control.held = True
            self.control.note(event)
            logger.info("HOLDING at home — send a task to start moving again")

        def home_action(self, robot) -> dict:
            return {k: self.control.home_pos for k in robot.action_features}

        def go_home(self, ctx) -> bool:
            """Interpolate to the home pose over ``home_duration_s``.

            NOT "send the final target and let the Δq clamp rate-limit it".
            That ramps at whatever the cap allows — 0.15 rad/tick at 30 Hz is
            ~4.5 rad/s — so the arm snapped home at the fastest speed the
            safety bound permitted, which reads as a jolt. Commanding the
            trajectory makes the speed a property of *this move* instead of a
            side effect of a limit; the clamp is still underneath, now as a
            backstop rather than the mechanism.

            Eased with smoothstep, so velocity starts and ends at zero. Peak is
            1.5x the average and still an order of magnitude under the clamp.

            Returns True if home was reached within tolerance.
            """
            robot = ctx.hardware.robot_wrapper
            cfg = ctx.runtime.cfg
            target = self.home_action(robot)
            dt = 1.0 / max(cfg.fps, 1.0)
            duration = max(self.control.home_duration_s, 0.0)

            def joint_err(obs) -> float:
                return max((abs(float(obs[k]) - target[k])
                            for k in target if k in obs), default=0.0)

            obs = robot.get_observation()
            if joint_err(obs) <= self.control.reset_tolerance:
                logger.info("already home")
                return True

            # Interpolate from where the arm actually is. A joint the
            # observation does not report is pinned to its target, i.e. left
            # alone by this ramp rather than swept from a guessed start.
            start = {k: (float(obs[k]) if k in obs else target[k]) for k in target}

            t_start = time.perf_counter()
            while (elapsed := time.perf_counter() - t_start) < duration:
                if ctx.runtime.shutdown_event.is_set():
                    return False
                tick = time.perf_counter()
                a = elapsed / duration
                a = a * a * (3.0 - 2.0 * a)              # smoothstep
                robot.send_action({k: start[k] + (target[k] - start[k]) * a
                                   for k in target})
                if (rest := dt - (time.perf_counter() - tick)) > 0:
                    precise_sleep(rest)

            # Settle. The arm lags the commanded waypoint, so hold the final
            # target until it actually arrives. With duration=0 this is the
            # whole move, i.e. the old clamp-limited behaviour.
            #
            # Always sends the exact target at least once: the ramp loop exits
            # with `a` a hair under 1.0, so its last waypoint is fractionally
            # short of home. Close enough to pass the tolerance check, which
            # would then skip the settle entirely and leave the arm parked at
            # not-quite-home for good.
            deadline = time.perf_counter() + self.control.reset_timeout_s
            while True:
                if ctx.runtime.shutdown_event.is_set():
                    return False
                tick = time.perf_counter()
                robot.send_action(target)
                err = joint_err(robot.get_observation())
                if err <= self.control.reset_tolerance or time.perf_counter() > deadline:
                    break
                if (rest := dt - (time.perf_counter() - tick)) > 0:
                    precise_sleep(rest)

            if err <= self.control.reset_tolerance:
                logger.info("home reached in %.1fs (max joint error %.4f rad)",
                            time.perf_counter() - t_start, err)
                return True
            logger.warning("home NOT reached — last error %.4f rad. The arm is "
                           "wherever the ramp got to; it was not teleported.", err)
            return False

        def reset_scene(self, ctx) -> None:
            """Ask the sim to put the props back, if the robot is a sim robot.

            Best-effort by design: on hardware there is nothing to reset (a
            real marker does not teleport home), and an older policy_server
            has no /reset route. Neither is an error worth aborting a reset
            for, so both are logged and stepped over.
            """
            robot = getattr(ctx.hardware.robot_wrapper, "inner", None)
            request = getattr(robot, "_request", None)
            if request is None:
                return
            try:
                out = request("/reset", {})
                logger.info("scene reset: %s", out.get("reset", "(no prop list)"))
            except Exception as e:
                logger.info("scene reset skipped (%s: %s)", type(e).__name__, e)

        def do_reset(self, ctx) -> None:
            """Home the arms, put the props back, clear the policy's queue."""
            logger.info("--- reset ---")
            self.go_home(ctx)
            self.reset_scene(ctx)
            self._engine.reset()
            if self._interpolator is not None:
                self._interpolator.reset()
            self._cached_obs_processed = None
            self._warmup_flushed = False
            with self.control._lock:
                self.control.resets += 1
                self.control.held = True
            logger.info("--- reset complete: parked, holding for a new task ---")

        def apply_commands(self, ctx) -> bool:
            """Drain the control queue. Returns False if asked to quit."""
            for cmd, arg in self.control.drain():
                if cmd == "task":
                    self.retarget(arg.strip())
                    self.control.note(f"task={arg.strip()!r}")
                elif cmd == "park":
                    # Server-side ramp, so an interactive park and an
                    # end-of-run park are the same code path.
                    robot = getattr(ctx.hardware.robot_wrapper, "inner", None)
                    request = getattr(robot, "_request", None)
                    if request is None:
                        logger.info("park: robot has no sim transport; "
                                    "using the local ramp instead")
                        self.go_home(ctx)
                    else:
                        try:
                            info = request("/park",
                                           {"duration_s": self.control.home_duration_s},
                                           timeout=30.0)
                            logger.info("parked (residual %.4f rad)",
                                        info.get("max_residual_rad", float("nan")))
                        except Exception as e:
                            logger.warning("park failed (%s: %s)",
                                           type(e).__name__, e)
                    # All three of park/reset/home put the arm somewhere safe on
                    # purpose, so all three hold it there. Stepping the policy
                    # again on the unchanged task would undo the move within one
                    # tick, which is what made these look like they did nothing.
                    self.hold("parked")
                elif cmd == "reset":
                    self.do_reset(ctx)          # sets the hold itself
                    self.control.note("reset")
                elif cmd == "home":
                    self.go_home(ctx)
                    self.hold("home")
                elif cmd == "pause":
                    with self.control._lock:
                        self.control.paused = True
                    logger.info("PAUSED -- policy is not being stepped")
                    self.control.note("paused")
                elif cmd == "resume":
                    # Resuming mid-chunk would replay stale actions computed
                    # from a pre-pause observation, so start clean.
                    self._engine.reset()
                    if self._interpolator is not None:
                        self._interpolator.reset()
                    self._cached_obs_processed = None
                    with self.control._lock:
                        self.control.paused = False
                        # Explicit operator "go", so it also lifts a hold — the
                        # escape hatch for continuing the current task after a
                        # reset without retyping it.
                        self.control.held = False
                    logger.info("RESUMED")
                    self.control.note("resumed")
                elif cmd == "quit":
                    logger.info("quit requested over control channel")
                    ctx.runtime.shutdown_event.set()
                    return False
            return True

        # -- loop -------------------------------------------------------
        def run(self, ctx) -> None:
            engine = self._engine
            cfg = ctx.runtime.cfg
            robot = ctx.hardware.robot_wrapper
            interpolator = self._interpolator
            control = self.control

            control_interval = interpolator.get_control_interval(cfg.fps)
            start_time = time.perf_counter()
            engine.resume()

            with control._lock:
                control.task = engine._task
            logger.info("interactive loop started -- task %r", engine._task)

            while not ctx.runtime.shutdown_event.is_set():
                loop_start = time.perf_counter()

                if not self.apply_commands(ctx):
                    break

                if cfg.duration > 0 and (loop_start - start_time) >= cfg.duration:
                    logger.info("Duration limit reached (%.0fs)", cfg.duration)
                    break

                # Keep pulling observations while stopped: the camera feed stays
                # live in the visualiser, which is the point -- you are looking
                # at the scene while you reposition something, or deciding what
                # to send next while the arm holds at home.
                obs = robot.get_observation()
                # Publish before the check: a stopped loop is exactly when a
                # watcher most needs to see the scene.
                control.publish(obs, every=self.publish_every)

                if not control.stepping:
                    if (rest := control_interval - (time.perf_counter() - loop_start)) > 0:
                        precise_sleep(rest)
                    continue

                obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    continue

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                with control._lock:
                    control.ticks += 1

                dt = time.perf_counter() - loop_start
                if (rest := control_interval - dt) > 0:
                    precise_sleep(rest)
                else:
                    logger.warning(
                        "loop running slower (%.1f Hz) than target %.0f Hz",
                        1 / dt, cfg.fps)

    return InteractiveStrategy


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _extract_own_flags(argv: list[str]) -> tuple[dict, list[str]]:
    """Pull this module's own flags out of argv before draccus sees it.

    LeRobot parses with draccus, which errors on anything it does not
    recognise. Rather than fork its config dataclass, the extra flags are
    removed here and everything else is passed through verbatim -- so any
    ``lerobot-rollout`` command line works unchanged with the binary swapped.
    """
    own = {"--control-port": ("control_port", int),
           "--control-host": ("control_host", str),
           "--home": ("home", float),
           "--home-duration": ("home_duration", float),
           "--publish-every": ("publish_every", int)}
    out, rest, i = {}, [], 0
    while i < len(argv):
        tok = argv[i]
        key, _, inline = tok.partition("=")
        if key in own:
            name, cast = own[key]
            if inline:
                out[name] = cast(inline)
                i += 1
            else:
                if i + 1 >= len(argv):
                    raise SystemExit(f"{key} needs a value")
                out[name] = cast(argv[i + 1])
                i += 2
            continue
        rest.append(tok)
        i += 1
    return out, rest


def main() -> int:
    own, rest = _extract_own_flags(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]

    from lerobot.configs import parser
    from lerobot.rollout.configs import RolloutConfig
    from lerobot.rollout.context import build_rollout_context
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import (
        init_visualization,
        shutdown_visualization,
    )
    from lerobot.utils.import_utils import register_third_party_plugins

    # Registers yam_ultra_sim / yam_ultra_bimanual. Must run before the config
    # is parsed, or --robot.type is rejected as an unknown choice.
    register_third_party_plugins()
    InteractiveStrategy = build_strategy_class()

    def _run(cfg):
        init_logging()
        if cfg.display_data:
            init_visualization(cfg.display_mode, session_name="rollout",
                               ip=cfg.display_ip, port=cfg.display_port)

        signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
        ctx = build_rollout_context(cfg, signal_handler.shutdown_event)

        control = Control(
            home_pos=own.get("home", DEFAULT_HOME),
            home_duration_s=own.get("home_duration", Control.home_duration_s),
        )
        srv = _serve(control,
                     own.get("control_host", "127.0.0.1"),
                     own.get("control_port", DEFAULT_CONTROL_PORT))

        strategy = InteractiveStrategy(cfg.strategy, control)
        strategy.publish_every = max(1, int(own.get("publish_every", 1)))
        try:
            strategy.setup(ctx)
            strategy.run(ctx)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            srv.shutdown()
            strategy.teardown(ctx)
            if cfg.display_data:
                shutdown_visualization(cfg.display_mode)
        logger.info("Rollout finished")

    # parser.wrap() discovers the config class with
    # ``inspect.getfullargspec(fn).annotations[...]``, i.e. the RAW annotation.
    # This module has ``from __future__ import annotations``, so a written-out
    # ``cfg: RolloutConfig`` would arrive as the *string* "RolloutConfig" and
    # draccus would reject it with "must be called with a dataclass type".
    # Bind the real class instead of dropping the future-import for the whole
    # file.
    _run.__annotations__["cfg"] = RolloutConfig
    parser.wrap()(_run)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
