"""LeRobot ``Robot`` adapter for the bimanual YAM-Ultra.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs

try:
    # lerobot >=0.5: RobotAction/RobotObservation moved to lerobot.types.
    from lerobot.types import RobotAction, RobotObservation
except ImportError:
    # lerobot 0.4.x (still what the live-hardware `robot` conda env runs).
    from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected
from lerobot.utils.errors import DeviceNotConnectedError

from .arm_client import YamArmClient

logger = logging.getLogger(__name__)

ARM_JOINTS = 6
HANDS = ("left", "right")

# Most recently connect()ed instance, so BiQuestTeleoperator.connect() can
# auto-seed itself (see get_last_connected_follower() and its call site in
# teleop/bi_quest_teleop.py) — stock lerobot-record calls robot.connect()
# then teleop.connect() with no hook between them for this.
_LAST_CONNECTED: "YamUltraFollower | None" = None


def get_last_connected_follower() -> "YamUltraFollower | None":
    return _LAST_CONNECTED

_CAMERAS_YAML = Path(__file__).parent.parent / "cameras" / "config" / "cameras.yaml"


def _default_cameras() -> dict[str, CameraConfig]:
    """RealSense cameras wired from cameras/config/cameras.yaml (2 wrists + a
    third-person view — see that file for serials). Returns {} (no
    cameras) if the file is missing, so the follower still works headless.

    ``height`` here is the sensor's NATIVE capture height — what
    ``RealSenseCameraConfig``/``RealSenseCamera.connect()`` actually requests
    from the hardware, which must be a profile the sensor really offers (the
    D405 wrist cameras only do 640x480 at 640-wide; they do not have a 360
    mode at all). A separate, optional ``crop_height`` in the same yaml entry
    is NOT passed here — see ``_default_camera_crop_heights()`` — because
    cropping happens after capture, in ``_read_camera()``, not by asking the
    sensor for a resolution it cannot produce."""
    if not _CAMERAS_YAML.exists():
        logger.warning("%s not found — YamUltraFollower will have no default cameras",
                        _CAMERAS_YAML)
        return {}
    data = yaml.safe_load(_CAMERAS_YAML.read_text()) or {}
    cams: dict[str, CameraConfig] = {}
    for cam_id, cfg in (data.get("cameras") or {}).items():
        cams[cam_id] = RealSenseCameraConfig(
            serial_number_or_name=str(cfg["serial"]),
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30),
        )
    return cams


def _default_camera_crop_heights() -> dict[str, int]:
    """name -> crop_height for cameras whose yaml entry sets one. Paired with
    ``_default_cameras()`` (same file, same iteration) but kept separate
    since it's about post-capture processing, not what gets requested from
    the sensor. See ``YamUltraFollowerConfig.camera_crop_heights``."""
    if not _CAMERAS_YAML.exists():
        return {}
    data = yaml.safe_load(_CAMERAS_YAML.read_text()) or {}
    return {cam_id: int(cfg["crop_height"])
            for cam_id, cfg in (data.get("cameras") or {}).items()
            if "crop_height" in cfg}


def _center_crop_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    """Crop rows evenly off the top and bottom to reach ``target_h``.

    Used when a camera's dataset-declared height (to match a reference
    dataset's schema, e.g. MolmoAct2's 360-tall frames) is shorter than what
    the sensor can natively capture (e.g. the D405's 480, its only 640-wide
    option). A center crop preserves pixel scale and keeps the frame's
    vertical center in frame — no squash/stretch distortion. No-op (returns
    the frame unchanged) if ``target_h`` isn't smaller than the frame already
    is, so this is safe to call unconditionally."""
    h = frame.shape[0]
    if target_h >= h:
        return frame
    top = (h - target_h) // 2
    return frame[top: top + target_h]


@RobotConfig.register_subclass("yam_ultra_bimanual")
@dataclass
class YamUltraFollowerConfig(RobotConfig):
    """Config for the bimanual YAM-Ultra follower.

    ``--robot.type=yam_ultra_bimanual`` selects it in LeRobot CLIs once this
    module has been imported (importing it runs the registration above).
    """

    # For simulation purpose. For hardwares, these are defined in arm_server.py
    left_channel: str = "can_left"
    right_channel: str = "can_right"
    server_host: str = "localhost"
    left_server_port: int = 11333
    right_server_port: int = 11334
    sim: bool = False
    viewer: bool = False
    
    # Invert the gripper mapping on both arms (see module docstring).
    gripper_flip: bool = False

    # hardware safeguards
    max_joint_velocity: float | list[float] | None = field(
        default_factory=lambda: [4.0, 4.0, 4.0, 6.0, 6.0, 6.0] 
    )
    max_tick_s: float = 0.1
    max_relative_target: float = 0.15

    # Ramp the arms back to the pose they were in at connect()
    park_on_disconnect: bool = True
    park_duration_s: float = 5.0

    # NOTE: `camera_stale_grace_s` used to live here. _read_camera() no longer
    # reuses stale frames at all (a stale frame raises), so the field would
    # have been a --robot.* flag that silently did nothing. Removed rather
    # than left to lie about behaviour.

    # name -> CameraConfig (each must set width/height/fps, enforced by
    # RobotConfig.__post_init__). Names become observation.images.<name>.
    # Defaults to the rig's three RealSense cameras (cameras/config/cameras.yaml).
    cameras: dict[str, CameraConfig] = field(default_factory=_default_cameras)

    # name -> post-capture center-crop target height (see
    # _center_crop_height). Empty by default. Used when a camera's dataset-
    # declared height needs to be shorter than what the sensor can actually
    # capture — e.g. matching a reference dataset's video shape (MolmoAct2's
    # cameras are 640x360) when the D405 wrist cameras only support 640x480
    # natively. Doesn't apply to cameras that already capture at the target
    # height (the D435 top camera does 360 natively — no crop configured for
    # it in cameras.yaml, and this dict has no entry for it).
    camera_crop_heights: dict[str, int] = field(default_factory=_default_camera_crop_heights)


class YamUltraFollower(Robot):
    """Two i2rt YAM-Ultra arms + N cameras as one LeRobot Robot."""

    config_class = YamUltraFollowerConfig
    name = "yam_ultra_bimanual"

    def __init__(self, config: YamUltraFollowerConfig):
        super().__init__(config)
        self.config = config
        self._robots: dict = {}          # hand -> MotorChainRobot, filled on connect()
        self._n_dofs: dict[str, int] = {}
        self._connected = False
        self._last_clamp_warn: dict[str, float] = {}  # hand -> monotonic ts
        self._last_action_ts: float | None = None     # for the speed->Δq conversion
        # Last pose we commanded per hand. send_action seeds its target from
        # this so a joint the action omits holds instead of being driven, now
        # that we no longer read `present` follower-side before commanding.
        self._last_sent_qpos: dict[str, np.ndarray] = {}

        
        # hand -> (mjModel, mjData, viewer handle); see _open_viewers().
        self._viewer_state: dict | None = None
        
        # Only populated when config.sim spawns them — see _spawn_sim_servers.
        self._sim_servers: list = []

        # Built up-front (like SOFollower) so is_connected can poll them; they
        # don't open the device until .connect().
        self.cameras = make_cameras_from_configs(config.cameras)

    # ---- feature schema -----------------------------------------------------
    @property
    def _motors_ft(self) -> dict[str, type]:
        ft: dict[str, type] = {}
        for h in HANDS:
            for j in range(1, ARM_JOINTS + 1):
                ft[f"{h}_joint_{j}.pos"] = float
            ft[f"{h}_gripper.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # Declared height is post-crop when camera_crop_heights sets one for
        # this camera — must match what _read_camera() actually returns, or
        # the dataset writer's declared feature shape disagrees with the
        # array it's handed.
        return {
            name: (self.config.camera_crop_heights.get(name, self.config.cameras[name].height),
                   self.config.cameras[name].width, 3)
            for name in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ---- connection ---------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        # _arms_alive() matters here: i2rt's control loop can die at any point
        # (motor error / loss of communication) and nothing else would notice —
        # get_joint_pos() keeps returning the last state it read, so the robot
        # looks fine while every command goes nowhere.
        return (self._connected
                and self._arms_alive()
                and all(c.is_connected for c in self.cameras.values()))

    @property
    def n_dofs(self) -> dict[str, int]:
        """hand -> DoF count (6 arm-only, 7 with gripper). Populated on
        connect(); for callers driving this Follower directly (e.g. the
        hardware bridge scripts) that need to know if a gripper is present."""
        return dict(self._n_dofs)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # Cameras FIRST, arms second — order matters, don't swap it.
        # RealSense connect() is a burst of USB enumeration + bandwidth
        # reservation (~1s per device with warmup_s=1). The CAN adapters are
        # USB devices too, and i2rt's control loop is a free-running Python
        # thread polling motor feedback at 85-150 Hz with a short timeout.
        # Opening cameras while those loops run starves/disrupts the CAN
        # traffic long enough to trip "loss communication" on a random motor,
        # which makes i2rt set motor_chain.running=False and kill the control
        # thread — arms silently stop responding for the rest of the session.
        # Connecting cameras before the arms gets that burst over with while
        # nothing time-critical is running.
        #
        # Everything below runs under a failure guard: a half-finished connect
        # (one camera opened, then the next one EBUSY) would otherwise leave
        # those devices claimed by this process until it exits — the state that
        # makes every later run and `lerobot-find-cameras` report "device busy".
        self._connected = True  # so the cleanup path below is allowed to run
        try:
            for name, cam in self.cameras.items():
                logger.info("connecting camera %s", name)
                cam.connect()

            if self.config.sim:
                self._spawn_sim_servers()

            ports = {"left": self.config.left_server_port,
                     "right": self.config.right_server_port}
            for h in HANDS:
                logger.info("connecting to %s arm_server at %s:%d%s",
                            h, self.config.server_host, ports[h],
                            " (SIM)" if self.config.sim else "")
                self._robots[h] = YamArmClient(
                    self.config.server_host, ports[h],
                    # Refuse a stale sim server on a real run (and vice
                    # versa) — see YamArmClient for why that is otherwise
                    # silent, and warn if left/right look swapped.
                    expect_sim=self.config.sim,
                    expect_channel=(self.config.left_channel if h == "left"
                                    else self.config.right_channel),
                )
                self._n_dofs[h] = self._robots[h].num_dofs()

            self.configure()
            self._assert_arms_alive("connect")

            # Informational: park() targets ZERO (the arm's canonical rest
            # pose), not whatever it happens to be in now — so this is not a
            # stow target, it just tells you where the arms started. A pose
            # far from zero means they were not left folded, and the first
            # park will be a long move.
            start_qpos = {h: np.asarray(self._robots[h].get_joint_pos(), dtype=float)[:ARM_JOINTS]
                          for h in HANDS}
            for h in HANDS:
                logger.info("%s pose at connect: %s (park target is zero)",
                            h, np.round(start_qpos[h], 3).tolist())

            if self.config.viewer:
                self._open_viewers()
                if self._viewer_state is not None:
                    self._sync_viewers({f"{h}_joint_{j}.pos": float(start_qpos[h][j - 1])
                                         for h in HANDS for j in range(1, ARM_JOINTS + 1)})
        except BaseException:
            logger.warning("connect() failed — releasing whatever was already opened")
            try:
                self.disconnect()
            except Exception:
                logger.exception("cleanup after failed connect() also failed")
            self._connected = False
            raise

        global _LAST_CONNECTED
        _LAST_CONNECTED = self
        logger.info("%s connected (%d cameras).", self, len(self.cameras))

    def _open_viewers(self) -> None:
        """Two passive MuJoCo windows (left arm, right arm), this rig's own
        IK model (model/kinematics.py) — not i2rt's SimRobot, which has no
        window at all. Sets self._viewer_state to {hand: (model, data,
        viewer)} on success, or leaves it None (with a warning, never an
        exception) if there's no DISPLAY or MuJoCo can't open a window."""
        import os

        if not os.environ.get("DISPLAY"):
            logger.warning("config.viewer=True but no DISPLAY — skipping MuJoCo "
                            "viewer windows (run at the workstation, or DISPLAY=:0)")
            return
        try:
            import mujoco
            import mujoco.viewer

            from lerobot_robot_sparklab.robots.yam_ultra.model.kinematics import build_model_with_tool0_site

            state = {}
            for h in HANDS:
                vm, vd = build_model_with_tool0_site()
                vw = mujoco.viewer.launch_passive(vm, vd)
                state[h] = (vm, vd, vw)
            self._viewer_state = state
        except Exception:
            logger.exception("config.viewer=True but opening the MuJoCo viewer "
                              "windows failed — continuing without them")
            self._viewer_state = None

    def _sync_viewers(self, sent: dict[str, float]) -> None:
        """Push a just-sent (or just-captured) qpos into the viewer windows,
        if open. No-op if _open_viewers() never ran or failed. ``sent`` may
        cover only a subset of HANDS (e.g. park() stowing one arm) — hands
        missing from it are left showing their last-synced pose."""
        if self._viewer_state is None:
            return
        import mujoco

        for h in HANDS:
            if f"{h}_joint_1.pos" not in sent:
                continue
            vm, vd, vw = self._viewer_state[h]
            if not vw.is_running():
                continue
            vd.qpos[:ARM_JOINTS] = [sent[f"{h}_joint_{j}.pos"] for j in range(1, ARM_JOINTS + 1)]
            mujoco.mj_forward(vm, vd)
            vw.sync()

    def _close_viewers(self) -> None:
        # Doesn't call vw.close() — matches deploy/molmoact2.py's proven
        # pattern of just dropping the reference. launch_passive()'s render
        # thread is non-daemon, so a script holding these open won't exit on
        # its own until the windows are closed (by hand, or the process is
        # killed) regardless of whether close() is called — that's expected
        # passive-viewer behavior, not something disconnect() controls, and
        # harmless to leave open since this is only a debug aid.
        self._viewer_state = None

    def _spawn_sim_servers(self) -> None:
        """Start the two arm_servers ourselves, in --sim mode.

        Only for ``config.sim``: it keeps ``--robot.sim=true`` a single
        self-contained flag (tests, sim rollouts) instead of making every
        caller launch servers by hand. Real hardware deliberately does NOT
        do this — those servers own torque, outlive any one run, and park on
        their own exit, so their lifetime should not be tied to a follower.
        """
        import ctypes
        import signal
        import subprocess
        import sys

        def _die_with_parent() -> None:
            """PR_SET_PDEATHSIG: the kernel SIGTERMs this child when the
            parent dies, however it dies. ``disconnect()`` normally reaps
            them, but a hard kill (SIGKILL, crash, a test timing out) skips
            it and leaves the servers listening — orphaned sim servers then
            squat on the real ports and silently absorb the next hardware
            run, which the sim/real identity check now also catches. Belt
            and braces, because the failure is invisible."""
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)

        specs = [(self.config.left_channel, self.config.left_server_port),
                 (self.config.right_channel, self.config.right_server_port)]
        for channel, port in specs:
            logger.info("spawning SIM arm_server for %s on port %d", channel, port)
            self._sim_servers.append(subprocess.Popen(
                [sys.executable, "-m", "lerobot_robot_sparklab.robots.yam_ultra.arm_server",
                 "--channel", channel, "--port", str(port), "--sim",
                 # Nothing physical to drop, and skipping the ramp keeps
                 # teardown fast for tests.
                 "--no-park-on-exit"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=_die_with_parent,
            ))
        # YamArmClient.connect() retries for its connect_timeout_s, so there
        # is no sleep here — it absorbs the servers' startup.

    def _stop_sim_servers(self) -> None:
        for proc in self._sim_servers:
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    logger.exception("could not stop a spawned sim arm_server")
        self._sim_servers.clear()

    def _arms_alive(self) -> bool:
        """False once i2rt's control loop has died on any arm (motor error /
        loss of communication).

        The check now lives server-side (``ArmServer._alive``) and rides back
        on every read/command response, so this reads a cached flag rather
        than doing a round trip — see ``YamArmClient.is_alive``. Still worth
        checking on every I/O: a dead chain keeps answering ``get_joint_pos``
        with the last pose it read, so nothing else reveals it."""
        return all(robot.is_alive() for robot in self._robots.values())

    def _require_live(self, where: str) -> None:
        """Gate for the I/O methods, in place of ``@check_if_not_connected``.

        That decorator keys off ``is_connected``, which is now also False when
        the control loop has died — so it would report the misleading "not
        connected, run .connect() first" for a robot that connected fine and
        then lost CAN. Split the two cases so the message names the real fault.
        """
        if not self._connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__} is not connected. Run `.connect()` first."
            )
        self._assert_arms_alive(where)

    def _assert_arms_alive(self, where: str) -> None:
        if self._arms_alive():
            return
        dead = [h for h, r in self._robots.items() if not r.is_alive()]
        raise RuntimeError(
            f"{self}: i2rt control loop is dead on arm(s) {dead} (at {where}) — the "
            "motors stopped responding on CAN. Check the log above for the "
            "'loss communication' motor id, verify the E-stop is released and "
            "the arms are powered, then reconnect. Commanding this robot would "
            "do nothing while appearing to work."
        )

    # i2rt homes its own encoders on connect; there is no LeRobot-style
    # calibration file, so calibration is always considered satisfied.
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:  # no-op: handled inside i2rt on connect
        pass

    def configure(self) -> None:  # gains are set by get_yam_robot(); nothing to do
        pass

    # ---- I/O ----------------------------------------------------------------
    def get_observation(self) -> RobotObservation:
        self._require_live("get_observation")
        obs: dict[str, object] = {}
        # Issue both arms' reads before awaiting either — portal returns
        # futures, so this overlaps the two round trips instead of paying
        # them back to back.
        futs = {h: self._robots[h].read_async() for h in HANDS}
        for h in HANDS:
            q = np.asarray(self._robots[h].collect(futs[h])["pos"], dtype=float)
            for j in range(1, ARM_JOINTS + 1):
                obs[f"{h}_joint_{j}.pos"] = float(q[j - 1])
            if self._n_dofs[h] > ARM_JOINTS:
                g = float(q[ARM_JOINTS])
                obs[f"{h}_gripper.pos"] = 1.0 - g if self.config.gripper_flip else g
            else:
                obs[f"{h}_gripper.pos"] = 0.0

        for name, cam in self.cameras.items():
            obs[name] = self._read_camera(name, cam)

        return obs

    # 500ms is a good default, higher will cause bad frame mismatch, which will directly affect causality. This can be one line + cropping
    def _read_camera(self, name: str, cam) -> np.ndarray:
        """One camera frame. A stale frame raises — it is never reused.

        This used to fall back to the last good frame through a short hiccup,
        to stop a momentary stall from killing a long teleop recording
        session. That tolerance is gone deliberately:

        * It is wrong for a policy rollout. LeRobot's ``read_latest()`` raises
          past its own 500 ms bound for a reason — silently handing the policy
          a stale image makes it act on a scene that no longer exists, and the
          failure is invisible in the logs.
        * Its main justification no longer applies. The stalls it covered were
          largely CAN retry storms blocking ``send_action`` in this process;
          the arms now run in their own ``arm_server`` processes, so that
          source of stalls is gone.

        If a camera genuinely cannot keep up, that is worth failing on rather
        than papering over.
        """
        frame = cam.read_latest()

        crop_h = self.config.camera_crop_heights.get(name)
        if crop_h is not None:
            frame = _center_crop_height(frame, crop_h)
        return frame

    # Bounds on the measured tick interval used to turn max_joint_velocity
    # into a per-tick allowance. Without them, the first tick (no previous
    # timestamp) or a stall (episode reset, disk flush, a breakpoint) would
    # hand out a huge one-tick budget and let the arm lunge — the exact snap
    # the clamp exists to prevent. 0.1 s ≈ the slowest sane control tick.
    _MIN_TICK_S = 0.002
    # Upper bound is configurable — see YamUltraFollowerConfig.max_tick_s for
    # why it also throttles a slow loop, not just a stall.

    def _dq_caps(self) -> np.ndarray | None:
        """Per-joint Δq allowance (rad) for this tick, or None if uncapped.

        Combines the loop-rate-independent speed cap (max_joint_velocity ×
        measured tick interval) with the raw per-tick cap
        (max_relative_target); when both are set the tighter wins per joint.
        """
        now = time.monotonic()
        prev = self._last_action_ts
        self._last_action_ts = now
        # First tick has no interval to measure: assume the fastest rate, i.e.
        # the tightest budget, so an unseeded first action can't snap. Also the
        # path taken on the first tick after park() (which clears the
        # timestamp) — see park()'s finally block.
        if prev is None:
            dt = self._MIN_TICK_S
        else:
            # Budget from the measured gap.
            #
            # This was briefly changed to min(measured, median-of-recent) to
            # stop a stall from buying extra motion — a stall let the arm
            # earn budget while standing still, then spend it in one tick
            # (measured: 1.6 rad / 92 deg in a single tick, 2x the configured
            # 16 rad/s). That is a real effect, but the cure was worse:
            #
            # a chunked-policy loop is BIMODAL — ~29 cheap queue-serving
            # ticks, then one long inference tick. The median is therefore
            # the *cheap* interval, so the long tick got throttled to the
            # cheap budget: 0.02 rad instead of 0.4 rad, 20x less, on
            # exactly the tick where a fresh chunk lands and the policy wants
            # its largest move. On hardware the arm barely moved. (On a
            # uniformly-slow loop the two agree, which is why steady-rate
            # testing missed it.)
            #
            # The underlying problem is that this is a position-STEP cap
            # pretending to be a velocity cap: nothing here bounds how fast
            # the motor closes the step — that is the PD gains (kp=80). No
            # choice of dt fixes both ends. The real fix is to stop issuing
            # one big step at all: let the arm_server, which already runs at
            # ~88 Hz, interpolate each commanded step across the tick instead
            # of handing the PD loop a jump. Until then this errs toward
            # "the arm can keep up", because an arm that cannot track the
            # policy is useless, while the snap is bounded by _MAX_TICK_S.
            dt = now - prev
        dt = min(max(dt, self._MIN_TICK_S), self.config.max_tick_s)

        caps = None
        vel = self.config.max_joint_velocity
        if vel is not None:
            caps = np.broadcast_to(np.asarray(vel, dtype=float), (ARM_JOINTS,)) * dt

        mrt = self.config.max_relative_target
        if mrt is not None:
            per_tick = np.full(ARM_JOINTS, float(mrt))
            caps = per_tick if caps is None else np.minimum(caps, per_tick)
        return caps

    def send_action(self, action: RobotAction) -> RobotAction:
        """Command both arms. Returns the action actually sent (Δq-clamped),
        in the same action-space keys/units that were passed in."""
        # Stop the moment the control loop dies rather than logging a clamp
        # warning per tick forever: with a dead chain get_joint_pos() is frozen,
        # so `present` never advances, the teleop target runs away, and every
        # tick clamps against a stale pose while the arms sit still — and the
        # dataset would fill with frames whose actions were never executed.
        self._require_live("send_action")
        caps = self._dq_caps()
        sent: dict[str, float] = {}

        # One combined read+clamp+command per arm, both issued before either
        # is awaited: 2 round trips per tick instead of 4. The clamp itself is
        # unchanged — the follower still computes `caps` from ITS measured
        # tick interval and the server only applies them (see
        # ArmServer.command_clamped for why this halving matters).
        targets: dict[str, np.ndarray] = {}
        for h in HANDS:
            # Target defaults to "hold": any joint the action omits keeps its
            # last commanded value rather than being driven anywhere.
            present_hint = self._last_sent_qpos.get(h)
            cmd = (present_hint.copy() if present_hint is not None
                   else np.zeros(self._n_dofs[h]))

            for j in range(1, ARM_JOINTS + 1):
                key = f"{h}_joint_{j}.pos"
                if key in action:
                    cmd[j - 1] = float(action[key])

            if self._n_dofs[h] > ARM_JOINTS:
                gkey = f"{h}_gripper.pos"
                if gkey in action:
                    g = float(action[gkey])
                    cmd[ARM_JOINTS] = 1.0 - g if self.config.gripper_flip else g

            targets[h] = cmd

        # Fire both arms, then collect — overlapping the round trips.
        futs = {h: self._robots[h].command_clamped_async(targets[h], caps, ARM_JOINTS)
                for h in HANDS}
        for h in HANDS:
            out = self._robots[h].collect(futs[h])
            cmd = np.asarray(out["sent"], dtype=float)
            self._last_sent_qpos[h] = cmd.copy()

            overshoot = np.asarray(out.get("overshoot", np.zeros(ARM_JOINTS)), dtype=float)
            if overshoot.any():
                # Rate-limited per arm: sustained clamping is normal when the
                # teleop's own per-tick cap exceeds ours, and one line per tick
                # at the dataset FPS buries every other message in the log.
                now = time.monotonic()
                if now - self._last_clamp_warn.get(h, 0.0) > 1.0:
                    self._last_clamp_warn[h] = now
                    worst = int(np.argmax(overshoot))
                    logger.warning(
                        "%s action exceeded this tick's cap %.3f rad on joint %d by %.3f rad "
                        "— clamped (further messages for this arm suppressed for 1s)",
                        h, float(caps[worst]) if caps is not None else float("nan"),
                        worst + 1, float(overshoot[worst]))

            # Report back in action-space: arm joints as commanded, gripper
            # un-flipped to the 0..1 convention the caller used.
            for j in range(1, ARM_JOINTS + 1):
                sent[f"{h}_joint_{j}.pos"] = float(cmd[j - 1])
            if self._n_dofs[h] > ARM_JOINTS:
                gcmd = float(cmd[ARM_JOINTS])
                sent[f"{h}_gripper.pos"] = 1.0 - gcmd if self.config.gripper_flip else gcmd
            else:
                sent[f"{h}_gripper.pos"] = float(action.get(f"{h}_gripper.pos", 0.0))

        self._sync_viewers(sent)
        return sent

    def park(self, hands: Sequence[str] | None = None, duration_s: float | None = None,
             rate_hz: float = 50.0) -> None:
        """Ramp arms to the zero pose and hold. Safe to cut torque there.

        Delegates to the arm servers, which use i2rt's own
        ``MotorChainRobot.move_joints()`` — that already interpolates in joint
        space, so the tick-by-tick ramp this used to run over RPC is gone.

        Targets zero rather than a pose captured at connect(). Zero is the
        arm's canonical rest configuration (i2rt's own joint-limit-violation
        message tells you to "move the arm to zero position and power cycle
        the robot"), whereas a captured pose is merely wherever the arm
        happened to be — so a run that ended mid-air used to make "park" ramp
        to mid-air and then cut torque.

        ``hands`` defaults to both; pass e.g. ``["left"]`` to stow one arm and
        leave the other under teleop control. Both ramps are started before
        either is awaited, so the arms still move together rather than in
        sequence.

        Bypasses send_action's Δq clamp deliberately — a slow, bounded move to
        a known-safe pose, not operator input.

        Blocking for ``duration_s``. No-op (with a warning) if the control loop
        is dead: commands would go nowhere, and pretending otherwise would hide
        the reason the arms drop.

        ``rate_hz`` is unused now (move_joints fixes its own step count) and is
        kept only so existing callers don't break.
        """
        hands = tuple(HANDS if hands is None else hands)
        if not self._robots:
            logger.warning("park(): no arms connected — skipping")
            return
        if not self._arms_alive():
            logger.warning("park(): i2rt control loop is dead — cannot park, arms will "
                           "drop when torque is cut. Support them by hand.")
            return

        duration_s = self.config.park_duration_s if duration_s is None else duration_s
        # Log WHO asked. park() drives the arms to zero outside the Δq clamp,
        # so an unexpected call looks exactly like the arm yanking itself
        # home mid-run. The legitimate callers are disconnect(),
        # BiQuestTeleoperator._stow_arm (thumbstick / B-Y), and its idle
        # auto-stow (opt-in, auto_stow_idle_s). Anything else in this trace
        # is the bug.
        import traceback
        caller = " <- ".join(
            f"{fr.filename.rsplit('/', 1)[-1]}:{fr.lineno}({fr.name})"
            for fr in reversed(traceback.extract_stack()[-4:-1])
        )
        logger.info("parking %s arm(s): ramping to zero over %.1fs ... [called by %s]",
                    "+".join(hands), duration_s, caller)
        try:
            # Start both before awaiting either: the call blocks server-side
            # for the whole ramp, so awaiting one first would park serially.
            futures = {h: self._robots[h].park_async(duration_s) for h in hands}
            for h, fut in futures.items():
                if not bool(fut.result(timeout=duration_s + 30.0)["ok"]):
                    logger.warning("park(): %s arm could not park (control loop dead?) — "
                                   "it will drop when torque is cut", h)
        except Exception:
            logger.exception("park failed — arms may drop when torque is cut")
            return
        finally:
            # The ramp drove the arms without going through send_action(), so
            # the Δq clamp's clock is stale; clearing it puts the next tick on
            # the tightest-budget path instead of crediting the whole park.
            self._last_action_ts = None
            # park moved the arms behind send_action's back; drop the cached
            # hold-target so the next tick re-seeds from the real pose.
            self._last_sent_qpos.clear()

        # The ramp ran server-side, so the viewer could not be animated during
        # it — mirror the pose it ended at.
        try:
            self._sync_viewers({
                f"{h}_joint_{j}.pos": float(self._robots[h].get_joint_pos()[j - 1])
                for h in hands for j in range(1, ARM_JOINTS + 1)
            })
        except Exception:
            logger.debug("park(): viewer sync after ramp failed", exc_info=True)
        logger.info("parked %s at zero pose.", "+".join(hands))

    def disconnect(self) -> None:
        """Release both arms and every camera.

        Deliberately NOT guarded by ``@check_if_not_connected``: ``is_connected``
        goes False the moment i2rt's control loop dies, and that is exactly when
        cleanup matters most — a refused disconnect would leave the RealSense
        devices claimed by this process until it exits (they stay busy for any
        other tool, e.g. `lerobot-find-cameras`). Each step is independently
        guarded so one failure can't skip the rest.
        """
        if not self._connected:
            return

        # Park BEFORE close(): i2rt's close() cuts torque where the arm stands,
        # so an un-parked arm falls under gravity the moment the process exits.
        if self.config.park_on_disconnect:
            try:
                self.park()
            except Exception:
                logger.exception("park before disconnect failed — arms may drop on close")

        for h in HANDS:
            robot = self._robots.get(h)
            if robot is None:
                continue
            try:
                robot.close()
            except Exception:
                logger.exception("error closing %s arm", h)
        for name, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception:
                logger.exception("error disconnecting camera %s", name)
        self._close_viewers()
        # Last: the clients above are talking to these. Only ever non-empty
        # in sim — real servers are externally managed and outlive the run.
        self._stop_sim_servers()
        self._connected = False
        logger.info("%s disconnected.", self)
