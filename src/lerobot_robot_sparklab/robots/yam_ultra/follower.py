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

# Most recently connected instance: lerobot-record offers no hook between
# robot.connect() and teleop.connect(), so the teleop seeds itself from this.
_LAST_CONNECTED: "YamUltraFollower | None" = None


def get_last_connected_follower() -> "YamUltraFollower | None":
    return _LAST_CONNECTED

# Wrong path fails quietly: _default_cameras() returns {} and the follower comes
# up with no cameras, so a policy just receives an observation without them.
_CAMERAS_YAML = Path(__file__).parent / "config" / "cameras.yaml"


def _default_cameras() -> dict[str, CameraConfig]:
    """RealSense cameras from config/cameras.yaml, or {} if the file is missing.

    The yaml's ``height`` is the sensor's native capture height and must be a
    profile it really offers. Its optional ``crop_height`` is applied after
    capture in ``_read_camera()``, not here.
    """
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
    """name -> crop_height for cameras whose yaml entry sets one."""
    if not _CAMERAS_YAML.exists():
        return {}
    data = yaml.safe_load(_CAMERAS_YAML.read_text()) or {}
    return {cam_id: int(cfg["crop_height"])
            for cam_id, cfg in (data.get("cameras") or {}).items()
            if "crop_height" in cfg}


def _center_crop_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    """Crop rows evenly off top and bottom to reach ``target_h``.

    Matches a reference dataset's shorter frames without rescaling, preserving
    pixel scale. No-op if ``target_h`` is not smaller.
    """
    h = frame.shape[0]
    if target_h >= h:
        return frame
    top = (h - target_h) // 2
    return frame[top: top + target_h]


@RobotConfig.register_subclass("yam_ultra_bimanual")
@dataclass
class YamUltraFollowerConfig(RobotConfig):
    """``--robot.type=yam_ultra_bimanual`` in any LeRobot CLI."""

    # Sim only; on hardware these are set on the arm_server command line.
    left_channel: str = "can_left"
    right_channel: str = "can_right"
    server_host: str = "localhost"
    left_server_port: int = 11333
    right_server_port: int = 11334
    sim: bool = False
    
    # Invert the gripper mapping on both arms.
    gripper_flip: bool = False

    max_relative_target: float | list[float] | None = field(
        default_factory=lambda: [0.133, 0.133, 0.133, 0.15, 0.15, 0.15]
    )

    # Ramp the arms to the zero pose on disconnect, where torque is safe to cut.
    park_on_disconnect: bool = True
    park_duration_s: float = 5.0

    cameras: dict[str, CameraConfig] = field(default_factory=_default_cameras)
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
        # send_action seeds from this, so a joint the action omits holds.
        self._last_sent_qpos: dict[str, np.ndarray] = {}

        # Per-tick Δq cap, resolved once. None means the caller clamps itself.
        mrt = config.max_relative_target
        self._caps: np.ndarray | None = (
            None if mrt is None
            else np.broadcast_to(np.asarray(mrt, dtype=float),
                                 (ARM_JOINTS,)).copy())

        # Only populated when config.sim spawns them — see _spawn_sim_servers.
        self._sim_servers: list = []

        # Built up-front so is_connected can poll them; no device opened yet.
        self.cameras = make_cameras_from_configs(config.cameras)

    # In the arm's own position-vector order; shared by the three marshalling
    # sites below.
    _ARM_KEYS = {h: tuple(f"{h}_joint_{j}.pos" for j in range(1, ARM_JOINTS + 1))
                 for h in HANDS}
    _GRIPPER_KEYS = {h: f"{h}_gripper.pos" for h in HANDS}

    # ---- feature schema -----------------------------------------------------
    @property
    def _motors_ft(self) -> dict[str, type]:
        ft: dict[str, type] = {}
        for h in HANDS:
            ft.update(dict.fromkeys(self._ARM_KEYS[h], float))
            ft[f"{h}_gripper.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # Post-crop, to match what _read_camera() actually returns.
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
        # _arms_alive() matters: a dead control loop keeps returning the last
        # pose read, so the robot looks fine while commands go nowhere.
        return (self._connected
                and self._arms_alive()
                and all(c.is_connected for c in self.cameras.values()))

    @property
    def n_dofs(self) -> dict[str, int]:
        """hand -> DoF count (6 arm-only, 7 with gripper), after connect()."""
        return dict(self._n_dofs)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
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
                    expect_sim=self.config.sim,
                    expect_channel=(self.config.left_channel if h == "left"
                                    else self.config.right_channel),
                )
                self._n_dofs[h] = self._robots[h].num_dofs()

            self.configure()
            self._assert_arms_alive("connect")

            # A pose far from zero means the arms were not left folded, so the
            # first park will be a long move.
            start_qpos = {h: np.asarray(self._robots[h].get_joint_pos(), dtype=float)[:ARM_JOINTS]
                          for h in HANDS}
            for h in HANDS:
                logger.info("%s pose at connect: %s (park target is zero)",
                            h, np.round(start_qpos[h], 3).tolist())
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

    def _spawn_sim_servers(self) -> None:
        """Start both arm_servers in --sim mode, so ``--robot.sim=true`` is one
        self-contained flag.

        Hardware deliberately does not: those servers own torque, outlive any
        one run, and park on their own exit.
        """
        import ctypes
        import signal
        import subprocess
        import sys

        def _die_with_parent() -> None:
            """PR_SET_PDEATHSIG, so a hard kill of the parent cannot leave orphans
            squatting on the real ports and absorbing the next hardware run."""
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)

        specs = [(self.config.left_channel, self.config.left_server_port),
                 (self.config.right_channel, self.config.right_server_port)]
        for channel, port in specs:
            logger.info("spawning SIM arm_server for %s on port %d", channel, port)
            self._sim_servers.append(subprocess.Popen(
                [sys.executable, "-m", "lerobot_robot_sparklab.robots.yam_ultra.arm_server",
                 "--channel", channel, "--port", str(port), "--sim",
                 # Nothing physical to drop; skipping the ramp speeds teardown.
                 "--no-park-on-exit"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=_die_with_parent,
            ))
        # No sleep here: YamArmClient.connect() retries, absorbing the startup.

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
        """False once i2rt's control loop has died on any arm.

        Reads a cached flag that rides back on every response, so it is free to
        check on every I/O — and worth it, since nothing else reveals a dead chain.
        """
        return all(robot.is_alive() for robot in self._robots.values())

    def _require_live(self, where: str) -> None:
        """Gate for the I/O methods, replacing ``@check_if_not_connected``.

        That decorator keys off ``is_connected``, which is also False on a dead
        control loop, so it would blame "not connected" for a lost CAN link.
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

    # i2rt homes its own encoders on connect; there is no calibration file.
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
        
        # Both issued before either is awaited, to overlap the round trips.
        futs = {h: self._robots[h].read_async() for h in HANDS}
        for h in HANDS:
            q = np.asarray(self._robots[h].collect(futs[h])["pos"], dtype=float)
            obs.update(zip(self._ARM_KEYS[h], q[:ARM_JOINTS].tolist()))
            if self._n_dofs[h] > ARM_JOINTS:
                g = float(q[ARM_JOINTS])
                obs[f"{h}_gripper.pos"] = 1.0 - g if self.config.gripper_flip else g
            else:
                obs[f"{h}_gripper.pos"] = 0.0

        for name, cam in self.cameras.items():
            obs[name] = self._read_camera(name, cam)

        return obs

    def _read_camera(self, name: str, cam) -> np.ndarray:
        """One camera frame, cropped if the rig config asks for it.

        A stale frame raises rather than being reused — handing a policy an
        image of a scene that no longer exists fails invisibly.
        """
        frame = cam.read_latest()

        crop_h = self.config.camera_crop_heights.get(name)
        if crop_h is not None:
            frame = _center_crop_height(frame, crop_h)
        return frame

    def _dq_caps(self) -> np.ndarray | None:
        """Per-joint Δq allowance (rad) for one tick, or None if uncapped."""
        return self._caps

    def _target_qpos(self, hand: str, action: RobotAction) -> np.ndarray:
        """One arm's requested pose, as its own absolute position vector.

        Marshalling only: the arm server clamps, against a pose it reads under
        the motor lock. The baseline is the last *clamped* value the server
        reported, so a clamped joint cannot accumulate a phantom offset.

        hand: "left" or "right".
        action: `.pos` keys; omitted joints hold their last commanded value.
        """
        last = self._last_sent_qpos.get(hand)
        cmd = last.copy() if last is not None else np.zeros(self._n_dofs[hand])

        for i, key in enumerate(self._ARM_KEYS[hand]):
            if key in action:
                cmd[i] = float(action[key])

        gripper_key = self._GRIPPER_KEYS[hand]
        if self._n_dofs[hand] > ARM_JOINTS and gripper_key in action:
            g = float(action[gripper_key])
            cmd[ARM_JOINTS] = 1.0 - g if self.config.gripper_flip else g
        return cmd

    def send_action(self, action: RobotAction) -> RobotAction:
        """Command both arms.

        Returns: the action actually sent, Δq-clamped, in the same keys and
        units that were passed in.
        """
        self._require_live("send_action")
        caps = self._dq_caps()
        sent: dict[str, float] = {}

        # One combined read+clamp+command per arm: 2 round trips, not 4.
        targets = {h: self._target_qpos(h, action) for h in HANDS}

        # Fire both, then collect, overlapping the round trips.
        futs = {h: self._robots[h].command_clamped_async(targets[h], caps, ARM_JOINTS)
                for h in HANDS}

        for h in HANDS:
            out = self._robots[h].collect(futs[h])
            cmd = np.asarray(out["sent"], dtype=float)
            self._last_sent_qpos[h] = cmd.copy()

            # Absent only against an arm_server predating the field. Not spelled
            # get(..., zeros(6)): that allocates every tick, key present or not.
            overshoot = out.get("overshoot")
            if overshoot is not None and np.any(overshoot):
                # Rate-limited: sustained clamping is normal when the teleop's
                # cap exceeds ours, and would bury every other log line.
                now = time.monotonic()
                if now - self._last_clamp_warn.get(h, 0.0) > 1.0:
                    self._last_clamp_warn[h] = now
                    worst = int(np.argmax(overshoot))
                    logger.warning(
                        "%s action exceeded the Δq cap %.3f rad on joint %d by %.3f rad "
                        "— clamped (further messages for this arm suppressed for 1s)",
                        h, float(caps[worst]) if caps is not None else float("nan"),
                        worst + 1, float(overshoot[worst]))

            # Back to action space, gripper un-flipped to the caller's 0..1.
            sent.update(zip(self._ARM_KEYS[h], cmd[:ARM_JOINTS].tolist()))
            gripper_key = self._GRIPPER_KEYS[h]
            if self._n_dofs[h] > ARM_JOINTS:
                gcmd = float(cmd[ARM_JOINTS])
                sent[gripper_key] = 1.0 - gcmd if self.config.gripper_flip else gcmd
            else:
                sent[gripper_key] = float(action.get(gripper_key, 0.0))

        return sent

    def park(self, hands: Sequence[str] | None = None, duration_s: float | None = None,
             rate_hz: float = 50.0) -> None:
        """Ramp arms to the zero pose and hold, where torque is safe to cut.

        Zero rather than a pose captured at connect(), which would park a run
        that ended mid-air to mid-air. Bypasses send_action's Δq clamp: this is
        a slow bounded move to a known-safe pose, not operator input. Blocks.

        hands: defaults to both; pass ["left"] to stow one and leave the other
            under teleop control. Both ramps start before either is awaited.
        duration_s: ramp length; defaults to the config value.
        rate_hz: unused, kept for callers. No-op if the control loop is dead.
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

        import traceback
        caller = " <- ".join(
            f"{fr.filename.rsplit('/', 1)[-1]}:{fr.lineno}({fr.name})"
            for fr in reversed(traceback.extract_stack()[-4:-1])
        )
        logger.info("parking %s arm(s): ramping to zero over %.1fs ... [called by %s]",
                    "+".join(hands), duration_s, caller)
        try:
            # The call blocks server-side for the whole ramp, so awaiting one
            # first would park the arms serially.
            futures = {h: self._robots[h].park_async(duration_s) for h in hands}
            for h, fut in futures.items():
                if not bool(fut.result(timeout=duration_s + 30.0)["ok"]):
                    logger.warning("park(): %s arm could not park (control loop dead?) — "
                                   "it will drop when torque is cut", h)
        except Exception:
            logger.exception("park failed — arms may drop when torque is cut")
            return
        finally:
            self._last_sent_qpos.clear()

        logger.info("parked %s at zero pose.", "+".join(hands))

    def disconnect(self) -> None:
        """Release both arms and every camera, each step guarded independently.

        Not decorated with ``@check_if_not_connected``: ``is_connected`` goes
        False the moment the control loop dies, which is when cleanup matters
        most — a refused disconnect leaves the cameras claimed.
        """
        if not self._connected:
            return

        # Before close(), which cuts torque wherever the arm stands.
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
        # Last: the clients above talk to these. Non-empty only in sim.
        self._stop_sim_servers()
        self._connected = False
        logger.info("%s disconnected.", self)
