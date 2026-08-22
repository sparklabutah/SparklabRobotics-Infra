"""The Isaac twin as a LeRobot Robot — ``--robot.type=yam_ultra_sim``.

Talks to ``sparklab_sim.policy_server`` over HTTP and presents the exact
observation/action schema the hardware follower does, so a policy rollout
runs against the simulator with no policy changes:

    # 1. Isaac, once. Leave it running across rollouts — it costs ~15 s to
    #    boot and holds the GPU.
    ./scripts/isaac_python.sh -m sparklab_sim.policy_server

    # 2. the same rollout CLI as hardware, with one flag changed
    lerobot-rollout --robot.type=yam_ultra_sim ... --policy.path=...

WHY THIS IS A SEPARATE ROBOT AND NOT ``--robot.sim=true``
That flag already exists on the hardware follower and does something
different: it spawns kinematic ``arm_server``s so the joints move with no CAN
bus. It does nothing about cameras — those stay real RealSense devices. A
policy needs images, so a rollout with no hardware needs a *rendered* rig,
which is this.

WHY HTTP AND NOT AN IMPORT
Isaac ships its own Python 3.12 with no lerobot, and importing
``lerobot_robot_sparklab`` under it fails at the package ``__init__``. The two
interpreters cannot be merged, so they talk over a socket — the same split the
arm servers already use.

WHAT IS AND IS NOT SIMULATED
Kinematic only. Joint targets are written straight through; nothing resists
them, nothing collides, and the gripper closes on nothing. A policy will
happily "grasp" through the box. This is the right tool for checking that a
checkpoint produces sane, in-range, temporally-coherent actions from real
camera geometry — and the wrong tool for judging whether a grasp succeeds.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from lerobot.cameras import CameraConfig  # noqa: F401  (schema parity)
from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig

logger = logging.getLogger(__name__)

HANDS = ("left", "right")
ARM_JOINTS = 6
# Fixed by the recorded dataset (meta/info.json): every image feature is
# 360x640x3. The server crops the wrist cameras to match.
IMAGE_SHAPE = (360, 640, 3)
CAMERA_NAMES = ("top", "left_wrist", "right_wrist")


@RobotConfig.register_subclass("yam_ultra_sim")
@dataclass
class YamUltraSimConfig(RobotConfig):
    """Config for the Isaac-backed bimanual YAM-Ultra."""

    host: str = "127.0.0.1"
    port: int = 8081

    # Per-request timeout. Generous: a cold render product or a stalled GPU
    # can take a while on the first tick, and failing a long rollout over one
    # slow frame is worse than waiting for it.
    timeout_s: float = 20.0

    # How long connect() waits for the server. Isaac takes ~15 s to boot from
    # cold, so a rollout started in the same breath as the server should not
    # give up immediately.
    connect_timeout_s: float = 90.0

    # Mirrors the hardware follower's flag so the same action dicts work.
    gripper_flip: bool = False

    # Ramp the arms back to the folded pose when the rollout disconnects.
    #
    # The sim server is a PERSISTENT workspace -- disconnect deliberately
    # leaves it running so the next rollout does not pay Isaac's boot cost --
    # which means whatever pose a run ends in is the pose the next run starts
    # from. LeRobot then captures that as its "initial position" and even
    # restores to it at teardown, so a single bad ending becomes permanent.
    # Measured on a run that ended without parking: the right arm sat 44 sigma
    # outside the pose any demo starts from, and the policy flailed.
    #
    # Parking here closes the loop for a clean exit. A killed or crashed
    # rollout never reaches this code -- the server's own idle auto-park
    # (--auto-park-s) is what covers that.
    park_on_disconnect: bool = True
    park_seconds: float = 2.5

    # ---- action shaping, deliberately mirroring the hardware follower ------
    # These are COPIES of YamUltraFollowerConfig's values, not a shared import.
    # The point of this robot is to predict what the arms would do, and an
    # unclamped sim silently flatters the policy: a chunk that would be cut
    # down to 0.15 rad on hardware executes in full here, so the very failure
    # you are hunting cannot reproduce.
    #
    # ** They are duplicated, so they WILL drift.** If you change a cap in
    # follower.py, change it here too. (A shared module was the alternative;
    # independent files were chosen to keep the hardware path untouched.)
    #
    # Per-joint Δq allowance for one tick, in radians; a scalar broadcasts,
    # None disables the clamp. A constant, which is also why this file no
    # longer needs nominal_tick_s or clamp_on_measured_tick: those existed only
    # to stop the sim's own slow wall-clock (~9.5 Hz against 30 Hz) from
    # inflating a velocity × dt budget by ~3x and flattering the policy. With
    # the cap expressed directly there is no dt to get wrong, and the sim
    # reproduces hardware's bound by construction.
    max_relative_target: float | list[float] | None = field(
        default_factory=lambda: [0.133, 0.133, 0.133, 0.15, 0.15, 0.15]
    )


class YamUltraSim(Robot):
    """Two simulated YAM-Ultra arms + three rendered cameras, as one Robot."""

    config_class = YamUltraSimConfig
    name = "yam_ultra_sim"

    def __init__(self, config: YamUltraSimConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._url = f"http://{config.host}:{config.port}"
        self._last_action: dict[str, float] | None = None
        # Observation returned by the last /step, waiting to be consumed by
        # get_observation(). See that method for why this halves the tick.
        self._pending_obs: dict | None = None
        # Latest joint state seen from the sim. send_action clamps against
        # this, mirroring arm_server.command_clamped clamping against the
        # arm's measured present pose.
        self._last_state: dict[str, float] | None = None
        self._last_clamp_warn: float = 0.0

        # Per-tick Δq cap, resolved once. See max_relative_target on the config.
        mrt = config.max_relative_target
        self._caps: np.ndarray | None = (
            None if mrt is None
            else np.broadcast_to(np.asarray(mrt, dtype=float),
                                 (ARM_JOINTS,)).copy())

    # ---- feature schema -----------------------------------------------------
    @property
    def _motors_ft(self) -> dict[str, type]:
        ft: dict[str, type] = {}
        for h in HANDS:
            for j in range(1, ARM_JOINTS + 1):
                ft[f"{h}_joint_{j}.pos"] = float
            ft[f"{h}_gripper.pos"] = float
        return ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **{n: IMAGE_SHAPE for n in CAMERA_NAMES}}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ---- transport ----------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None,
                 timeout: float | None = None) -> dict:
        """One request. ``timeout`` overrides the config default.

        Needed because /park deliberately blocks for the length of its ramp,
        which is longer than the per-tick timeout tuned for /step.
        """
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url + path, data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET")
        with urllib.request.urlopen(
                req, timeout=self.config.timeout_s if timeout is None else timeout) as r:
            body = json.loads(r.read())
        if "error" in body:
            raise RuntimeError(f"sim server: {body['error']}")
        return body

    @staticmethod
    def _decode(images: dict[str, str]) -> dict[str, np.ndarray]:
        import cv2

        out = {}
        for name, b64 in images.items():
            buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"could not decode image {name!r}")
            frame = bgr[..., ::-1]           # BGR -> RGB
            if frame.shape != IMAGE_SHAPE:
                # Loud, not coerced: a shape mismatch means the sim and the
                # dataset disagree about the camera, and silently resizing
                # would hand the policy a subtly wrong field of view.
                raise RuntimeError(
                    f"camera {name!r} returned {frame.shape}, expected "
                    f"{IMAGE_SHAPE} — check CAMERAS in policy_server.py")
            out[name] = np.ascontiguousarray(frame)
        return out

    # ---- lifecycle ----------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Nothing to calibrate: the twin is defined by the URDF."""

    def configure(self) -> None:
        """No gains to set — joint targets are written kinematically."""

    def connect(self, calibrate: bool = True) -> None:
        deadline = time.monotonic() + self.config.connect_timeout_s
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                health = self._request("/health")
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last = e
                time.sleep(1.0)
        else:
            raise ConnectionError(
                f"no sim server on {self._url} after "
                f"{self.config.connect_timeout_s:.0f}s ({last}). Start it with:\n"
                f"    ./scripts/isaac_python.sh -m sparklab_sim.policy_server")

        served = set(health.get("cameras", {}))
        if served != set(CAMERA_NAMES):
            raise RuntimeError(
                f"sim server serves cameras {sorted(served)}, this robot "
                f"expects {sorted(CAMERA_NAMES)} — the observation schema is "
                f"fixed by the recorded dataset")
        self._connected = True
        logger.info("%s connected to %s (cameras: %s)", self, self._url,
                    ", ".join(sorted(served)))

    def disconnect(self) -> None:
        # Deliberately does NOT stop the server. Isaac costs ~15 s to boot and
        # holds the GPU; keeping it warm across rollouts is the whole point of
        # running it as a separate process. But precisely BECAUSE it survives,
        # the pose left behind is inherited by the next run -- so hand it back
        # tidy.
        if self.config.park_on_disconnect and self._connected:
            try:
                # The server ramps this itself rather than us streaming the
                # trajectory: it owns the workspace, and a park must still
                # happen the same way when some other client asks for it.
                info = self._request(
                    "/park", {"duration_s": self.config.park_seconds},
                    timeout=self.config.timeout_s + self.config.park_seconds + 5.0)
                logger.info("%s parked arms (residual %.4f rad)", self,
                            info.get("max_residual_rad", float("nan")))
            except Exception as e:
                # Never let tidying up turn a finished rollout into a failure.
                logger.warning("%s could not park on disconnect (%s: %s); the "
                               "server's idle auto-park will catch it",
                               self, type(e).__name__, e)

        self._connected = False
        self._pending_obs = None
        self._last_state = None
        logger.info("%s disconnected (sim server left running).", self)

    # ---- control ------------------------------------------------------------
    def _require_live(self, where: str) -> None:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected ({where})")

    def _unpack(self, body: dict) -> dict:
        self._last_state = dict(body["state"])
        obs: dict[str, object] = dict(body["state"])
        obs.update(self._decode(body["images"]))
        return obs

    def get_observation(self) -> dict:
        """The observation from the most recent step, or a fresh render.

        ``/step`` already renders after applying the action, so consuming that
        result here is what makes a control tick ONE render instead of two.
        Without this the loop was measurably half as fast -- get_observation
        was throwing away a frame the server had just produced and asking for
        another one identical to it.

        The cache is consumed, not kept: two get_observation() calls without an
        intervening action must not return the same stale frame, because a
        policy reading twice is asking "has anything changed".
        """
        self._require_live("get_observation")
        if self._pending_obs is not None:
            body, self._pending_obs = self._pending_obs, None
            return self._unpack(body)
        return self._unpack(self._request("/obs"))

    # The 12 rate-limited arm-joint keys, in the order the clamp vector uses:
    # both hands, joints 1..6. Built once at class scope rather than
    # reformatted 12 times per tick. Grippers are deliberately absent -- they
    # are not rate-limited (see send_action).
    _ARM_KEYS = tuple(f"{h}_joint_{j}.pos"
                      for h in HANDS
                      for j in range(1, ARM_JOINTS + 1))

    def _dq_caps(self) -> np.ndarray | None:
        """Per-joint Δq allowance (rad) for one tick, or None if uncapped.

        Mirrors ``YamUltraFollower._dq_caps``: a constant, resolved in
        __init__ from max_relative_target.
        """
        return self._caps

    def send_action(self, action: dict) -> dict:
        """Apply an action and return what was actually sent (Δq-clamped).

        One HTTP round trip does act-then-render, because that is what a
        control tick is; the rendered frames land in the *next*
        get_observation, which is the same ordering the hardware follower has.
        """
        self._require_live("send_action")
        caps = self._dq_caps()

        # Clamp against the PRESENT pose, not the last command. That is what
        # arm_server.command_clamped does:
        #     step = target - present
        #     cmd  = present + clip(step, -caps, caps)
        # Clamping against the last command instead would let error accumulate
        # silently whenever the arm failed to reach it.
        present = self._last_state or {}
        last = self._last_action or {}

        # Hold the last commanded value for anything the action omits, rather
        # than defaulting to zero and snapping the arm home.
        target = np.fromiter(
            (float(action.get(k, last.get(k, 0.0))) for k in self._ARM_KEYS),
            dtype=float, count=len(self._ARM_KEYS))

        if caps is None:
            clamped = target
            worst = None
        else:
            # caps is per-joint (6,); the same six apply to each hand, so tile
            # rather than re-deriving them per key.
            cap = np.tile(caps, len(HANDS))
            p = np.fromiter(
                (float(present.get(k, t)) for k, t in zip(self._ARM_KEYS, target)),
                dtype=float, count=len(self._ARM_KEYS))
            step = target - p
            clamped = p + np.clip(step, -cap, cap)

            over = np.abs(step) - cap
            i = int(np.argmax(over))
            worst = ((float(over[i]), self._ARM_KEYS[i][:-len(".pos")])
                     if over[i] > 0 else None)

        sent: dict[str, float] = dict(zip(self._ARM_KEYS, clamped.tolist()))

        # The gripper is NOT rate-limited, matching command_clamped, which
        # only ever clamps indices [:n_arm]. It is a normalised 0..1 command
        # onto a small prismatic pair, not a link that can throw the arm
        # around.
        for h in HANDS:
            gk = f"{h}_gripper.pos"
            g = float(action.get(gk, last.get(gk, 0.0)))
            sent[gk] = 1.0 - g if self.config.gripper_flip else g

        if worst is not None:
            # Rate-limited exactly like the hardware follower's: sustained
            # clamping is normal when a policy's step exceeds our cap, and one
            # line per tick at dataset FPS buries every other message.
            now = time.monotonic()
            if now - self._last_clamp_warn > 1.0:
                self._last_clamp_warn = now
                logger.warning(
                    "sim action exceeded this tick's cap on %s by %.3f rad — "
                    "clamped. Hardware would clamp here too (further messages "
                    "suppressed for 1s)", worst[1], worst[0])

        self._pending_obs = self._request("/step", {"action": sent})
        if self._pending_obs is not None:
            self._last_state = dict(self._pending_obs["state"])
        self._last_action = dict(sent)
        return sent
