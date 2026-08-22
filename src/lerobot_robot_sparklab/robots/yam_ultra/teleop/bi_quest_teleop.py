"""Bimanual VR teleoperation via WebXR/Quest.

Reads `xr_frame` from the relay's /ws; per arm keeps a ClutchPoseMapper and a
DecoupledIKSolver. Per tick `_update_arm` runs: staleness gate -> EMA pose
filter -> precision button -> clutch edges -> IK step -> haptic mix.

get_action() returns {left,right}_joint_{1..6}.pos (rad) and
{left,right}_gripper.pos (0..1), matching YamUltraFollower.action_features.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "lerobot is required to use BiQuestTeleoperator. "
        "Run from a project environment with lerobot installed."
    ) from e

try:
    import websockets
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "websockets is required to use BiQuestTeleoperator. "
        "Install with: uv add websockets   (or pip install websockets)"
    ) from e

from ....core.pose_mapping import ClutchPoseMapper
from ..model.kinematics import DEFAULT_Q_REST
from ..ik.decoupled_ik import DecoupledIKSolver

logger = logging.getLogger(__name__)


# Default rotation taking Quest `local-floor` world axes into the arm base
DEFAULT_R_CALIB = [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

# WebXR xr-standard mapping: 0=trigger, 1=squeeze, 3=thumbstick, 4=A/X, 5=B/Y.
GRIP_BUTTON_INDEX = 1  # clutch
TRIGGER_BUTTON_INDEX = 0  # analog 0..1: gripper closure
PRECISION_BUTTON_INDEX = 4  # A/X — hold for precision scale
HANDOFF_BUTTON_INDEX = 5  # B/Y — intervention handoff
REST_RAMP_BUTTON_INDEX = 3  # thumbstick click — per-arm go-home ramp
ARM_DOFS = 6
NQ = 8  # arm (6) + 2 gripper-finger sliders

# Beyond this gap the controller stream is stale: force-disengage and refuse new
# engagements, so a WS that stalls then floods cannot cause catch-up motion.
XR_FRAME_STALE_TIMEOUT_S = 0.2

# Gripper prismatic range from the URDF; both fingers share this qpos scale.
GRIPPER_QPOS_OPEN = 0.001
GRIPPER_QPOS_CLOSED = -0.045


@TeleoperatorConfig.register_subclass("bi_quest_teleop")
@dataclass
class BiQuestTeleoperatorConfig(TeleoperatorConfig):
    """Config for BiQuestTeleoperator.

    `rest_qpos_{left,right}` (six joint angles, rad) seed each arm's qpos and
    bias the IK Tikhonov term, which is what breaks the elbow-flip ambiguity.
    """

    ws_url: str = "ws://127.0.0.1:8443/ws"
    publish_ik_state: bool = True
    connect_timeout_s: float = 5.0
    # Defaults to the vendored yam_ultra.xml; override for tightened ranges.
    arm_xml_path: str = ""
    # Row-major 3x3 taking Quest world vectors into the arm base frame.
    r_calib: list[list[float]] = field(default_factory=lambda: [row[:] for row in DEFAULT_R_CALIB])
    rest_qpos_left: list[float] = field(default_factory=lambda: DEFAULT_Q_REST.tolist())
    rest_qpos_right: list[float] = field(default_factory=lambda: DEFAULT_Q_REST.tolist())

    # IK / mapping knobs.
    lam: float = 0.05  # Position-solve base damping
    lam0: float = 0.15  # Adaptive-damping ramp amplitude near singular
    w0: float = 0.05  # Manipulability threshold where ramp starts
    mu: float = 0.02  # Tikhonov stiffness toward q_rest
    lam_rot: float = 0.05  # Orientation-solve base damping
    lam0_rot: float = 0.4  # Extra damping ramp amplitude near wrist gimbal
    w0_rot: float = 0.5  # Wrist-manipulability threshold where ramp starts
    
    rot_reach_limit: float = 0.6  # rad (~34°)
    pos_reach_limit: float = 0.25  # m
    rot_err_hold: float = 2.2
    scale_translation: float = 1.5  # controller→EE translation gain
    scale_rotation: float = 1.5  # controller→EE rotation gain (<1 = softer)
    precision_factor: float = 0.5

    # EMA on the controller pose. 1.0 = raw; at 200 Hz, 0.5 ≈ 7 ms time constant.
    pose_filter_alpha: float = 0.8
    max_dq_per_joint: list[float] | None = field(
        default_factory=lambda: [0.06, 0.06, 0.06, 0.24, 0.24, 0.24]
    )

    max_dq_per_joint_scalar_pos: float = 0.06
    max_dq_per_joint_scalar_rot: float = 0.24  # 4x position (48 rad/s @ 200 Hz)

    force_haptic_threshold_nm_left: float = 0.35
    force_haptic_threshold_nm_right: float = 0.35
    force_haptic_max_nm: float = 1.0
    force_haptic_velocity_comp_nm: float = 0.5
    force_haptic_enabled: bool = True

    # Go-home ramp: lerp qpos[:6] to rest_qpos_{hand} over this window.
    rest_ramp_duration_s: float = 2.0

    idle_resync_interval_s: float | None = 1.0
    auto_stow_idle_s: float | None = None


class BiQuestTeleoperator(Teleoperator):
    config_class = BiQuestTeleoperatorConfig
    name = "bi_quest_teleop"

    def __init__(self, config: BiQuestTeleoperatorConfig) -> None:
        super().__init__(config)
        self.config = config

        self._r_calib = np.asarray(config.r_calib, dtype=float)
        if self._r_calib.shape != (3, 3):
            raise ValueError(
                f"r_calib must be a 3x3 rotation matrix, got shape {self._r_calib.shape}"
            )

        self._lock = threading.RLock()
        self._latest_xr_frame: dict | None = None
        self._stow_follower = None
        self._idle_since: float | None = None       # monotonic ts both hands went disengaged
        self._last_idle_resync_t: float = 0.0
        self._auto_stowed_this_idle: bool = False    # avoid re-parking every tick while still idle
        self._last_xr_frame_time: float = 0.0

        rest_qpos = {
            "left": np.asarray(config.rest_qpos_left, dtype=float),
            "right": np.asarray(config.rest_qpos_right, dtype=float),
        }
        for hand, q in rest_qpos.items():
            if q.shape != (ARM_DOFS,):
                raise ValueError(f"rest_qpos_{hand} must have {ARM_DOFS} values, got shape {q.shape}")

        self._arms: dict[str, dict] = {}
        for hand in ("left", "right"):
            q_rest = rest_qpos[hand]
            qpos_init = np.zeros(NQ)
            qpos_init[:ARM_DOFS] = q_rest
            arm_solver = DecoupledIKSolver(
                arm_xml_path=config.arm_xml_path or None,
                lam_pos=config.lam,
                lam0=config.lam0,
                w0=config.w0,
                mu=config.mu,
                lam_rot=config.lam_rot,
                lam0_rot=config.lam0_rot,
                w0_rot=config.w0_rot,
                rot_err_hold=config.rot_err_hold,
                q_rest=q_rest.copy(),
                max_dq_per_joint=config.max_dq_per_joint,
            )
            self._arms[hand] = {
                "solver": arm_solver,
                "mapper": ClutchPoseMapper(
                    R=self._r_calib.copy(),
                    scale=config.scale_translation,
                    scale_rotation=config.scale_rotation,
                    rot_reach_limit=config.rot_reach_limit,
                    pos_reach_limit=config.pos_reach_limit,
                ),
                "qpos": qpos_init,
                "last_grip": False,
                "last_precision": False,  # tracks A/X button — re-anchor on edge
                "trigger": 0.0,
                "engaged": False,
                "haptic": 0.0,  # smoothed 0..1, broadcast in ik_state
                "force_haptic": 0.0,  
                "prev_gripper_pos": None,
                "prev_gripper_t": None,
                "gripper_vel_filt": 0.0,
                "needs_reanchor": False,  # set during stale → re-anchor on recovery
                "pos_filt": None,  # EMA-smoothed controller position
                "quat_filt": None,  # EMA-smoothed controller orientation (wxyz)
                "last_rest_button": False,
                "ramp_active": False,
                "ramp_start_q": np.zeros(ARM_DOFS),
                "ramp_target_q": q_rest.copy(),
                "ramp_start_t": 0.0,
            }
        self._buttons: dict[str, dict[str, bool]] = {
            "left": {"grip": False, "handoff": False},
            "right": {"grip": False, "handoff": False},
        }

        # WS plumbing.
        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_stop: threading.Event | None = None
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._ws_connected = threading.Event()

        # Own timer so the resync runs even when get_action() does not — DAgger's
        # autonomous phase would otherwise leave qpos frozen and stale at handoff.
        self._idle_resync_thread: threading.Thread | None = None
        self._idle_resync_stop: threading.Event | None = None

        self._haptic_calib: dict | None = None

        # EMA-smoothed get_action() rate, published in ik_state so the web UI can
        # show the Δq-cap slider in honest rad/s.
        self._last_get_action_t: float | None = None
        self._loop_hz: float | None = None

    # ---------- Teleoperator interface ----------

    @property
    def action_features(self) -> dict[str, type]:
        feats: dict[str, type] = {}
        for hand in ("left", "right"):
            for j in range(1, ARM_DOFS + 1):
                feats[f"{hand}_joint_{j}.pos"] = float
            feats[f"{hand}_gripper.pos"] = float
        return feats

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._ws_connected.is_set()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        self._ws_stop = threading.Event()
        self._ws_thread = threading.Thread(
            target=self._ws_thread_main, name="bi-quest-teleop-ws", daemon=True
        )
        self._ws_thread.start()
        if not self._ws_connected.wait(timeout=self.config.connect_timeout_s):
            raise RuntimeError(f"BiQuestTeleoperator: timed out connecting to {self.config.ws_url}")
        logger.info("BiQuestTeleoperator connected to %s", self.config.ws_url)

        # Best-effort seed from the last connected follower so the first action
        # doesn't target rest_qpos. Local import: avoids depending on Robot here.
        from ..follower import get_last_connected_follower

        follower = get_last_connected_follower()
        if follower is not None:
            self.seed_qpos_from_obs(follower.get_observation())
            logger.info("BiQuestTeleoperator: seeded from %s's current pose", follower)
            # Routes the thumbstick go-home ramp through the follower too.
            self._stow_follower = follower

        self._idle_resync_stop = threading.Event()
        self._idle_resync_thread = threading.Thread(
            target=self._idle_resync_loop, name="bi-quest-teleop-idle-resync", daemon=True
        )
        self._idle_resync_thread.start()

    def disconnect(self) -> None:
        if self._idle_resync_stop is not None:
            self._idle_resync_stop.set()
        if self._idle_resync_thread is not None:
            self._idle_resync_thread.join(timeout=2.0)
        if self._ws_stop is not None:
            self._ws_stop.set()
        if self._ws_loop is not None and self._ws is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._ws.close(), self._ws_loop)
                fut.result(timeout=1.0)
            except Exception:
                pass
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=2.0)
        self._ws_connected.clear()
        logger.debug("BiQuestTeleoperator disconnected")

    def send_feedback(self, feedback: dict) -> None:
        """Push orchestrator feedback into per-arm haptic state.

        feedback: may carry `torques` with `{left,right}_gripper.torque`.
            Unrecognised keys are ignored.
        """
        if not isinstance(feedback, dict):
            return
        torques = feedback.get("torques")
        if isinstance(torques, dict):
            self._apply_torque_feedback(torques)

    def publish_state(self) -> None:
        """Force one ik_state broadcast without waiting for get_action()."""
        if self.config.publish_ik_state and self._ws is not None and self._ws_loop is not None:
            self._publish_ik_state_async()

    def _apply_torque_feedback(self, torques: dict) -> None:
        """Map gripper torque to 0..1 haptic intensity: θ_eff = θ_base + k_v·|v|.

        The velocity term masks the inertial spike during fast opens and closes.

        torques: `{left,right}_gripper.torque` (Nm), optionally `.pos` for the
            numerical velocity. The first tick after connect uses θ_base.
        """
        ceiling = max(1e-6, float(self.config.force_haptic_max_nm))
        kv = max(0.0, float(self.config.force_haptic_velocity_comp_nm))
        # EMA on the velocity so the threshold doesn't judder tick to tick.
        vel_alpha = 0.4
        now = time.time()
        with self._lock:
            for hand in ("left", "right"):
                tau_key = f"{hand}_gripper.torque"
                pos_key = f"{hand}_gripper.pos"
                if tau_key not in torques:
                    continue
                tau = abs(float(torques[tau_key]))
                arm = self._arms[hand]
                # Numerical velocity from successive pos samples.
                if pos_key in torques:
                    pos = float(torques[pos_key])
                    prev_pos = arm["prev_gripper_pos"]
                    prev_t = arm["prev_gripper_t"]
                    if prev_pos is not None and prev_t is not None:
                        dt = max(now - prev_t, 1e-3)
                        v_raw = (pos - prev_pos) / dt
                        arm["gripper_vel_filt"] = (1.0 - vel_alpha) * arm[
                            "gripper_vel_filt"
                        ] + vel_alpha * v_raw
                    arm["prev_gripper_pos"] = pos
                    arm["prev_gripper_t"] = now
                # Per-arm: static holding torque differs between grippers.
                threshold_base = max(
                    0.0, float(getattr(self.config, f"force_haptic_threshold_nm_{hand}"))
                )
                threshold_eff = threshold_base + kv * abs(arm["gripper_vel_filt"])
                # No dynamic range left — force 0 rather than saturating to max.
                if threshold_eff >= ceiling:
                    intensity = 0.0
                else:
                    span = ceiling - threshold_eff
                    intensity = max(0.0, min(1.0, (tau - threshold_eff) / span))
                # Still computed when disabled, so the toggle re-lights instantly.
                if not self.config.force_haptic_enabled:
                    intensity = 0.0
                arm["force_haptic"] = intensity
                # Track per-arm peak |τ| for threshold = peak + margin on close.
                if self._haptic_calib is not None:
                    self._haptic_calib["peak"][hand] = max(
                        self._haptic_calib["peak"][hand], tau
                    )
            # After the per-hand loop, so both arms get the same final sample.
            if self._haptic_calib is not None:
                calib = self._haptic_calib
                if now - calib["start_t"] >= calib["duration_s"]:
                    self._finalize_haptic_calibration()

    # ---------- Intervention (handoff) hooks ----------
    # Level, not edge. Reads `self._buttons`, which the WS reader keeps fresh.

    def is_engaged(self) -> bool:
        with self._lock:
            return self._buttons["left"]["grip"] or self._buttons["right"]["grip"]

    def is_handoff_pressed(self) -> bool:
        with self._lock:
            return self._buttons["left"]["handoff"] or self._buttons["right"]["handoff"]

    def is_pause_pressed(self) -> bool:
        """Right-hand B — pause/resume the policy. Level only."""
        with self._lock:
            return self._buttons["right"]["handoff"]

    def is_reverse_pressed(self) -> bool:
        """Left-hand Y — reverse, e.g. replay buffered actions backward. Level only."""
        with self._lock:
            return self._buttons["left"]["handoff"]

    def seed_qpos_from_obs(self, obs: dict[str, float]) -> None:
        """Reset qpos / gripper / engagement to match a `.pos`-keyed dict.

        Also force-disengages, so a grip released while nothing was calling
        get_action() cannot leave a stale mapper anchor behind.

        obs: robot observation or commanded action. Prefer the command at
            handoff — it keeps a gripper squeezing an object closed rather than
            recording its blocked-open measurement. Missing keys are skipped.
        """
        with self._lock:
            for hand in ("left", "right"):
                arm = self._arms[hand]
                for j in range(ARM_DOFS):
                    key = f"{hand}_joint_{j + 1}.pos"
                    if key in obs:
                        arm["qpos"][j] = float(obs[key])
                # Stored as `trigger` and mirrored into the sim-viewer qpos slots.
                gkey = f"{hand}_gripper.pos"
                if gkey in obs:
                    g = float(np.clip(obs[gkey], 0.0, 1.0))
                    arm["trigger"] = g
                    grip_qpos = GRIPPER_QPOS_OPEN + g * (GRIPPER_QPOS_CLOSED - GRIPPER_QPOS_OPEN)
                    arm["qpos"][6] = grip_qpos
                    arm["qpos"][7] = grip_qpos
                # Cancel any in-flight ramp: unguarded, it would overwrite the seed.
                # `last_rest_button` is left alone so a held thumbstick can't re-arm it.
                arm["ramp_active"] = False
                arm["mapper"].disengage()
                arm["engaged"] = False
                arm["last_grip"] = False
                arm["pos_filt"] = None
                arm["quat_filt"] = None
                arm["needs_reanchor"] = False

    def _idle_resync_loop(self) -> None:
        """Poll _check_idle_resync_and_autostow() regardless of get_action() calls."""
        stop = self._idle_resync_stop
        assert stop is not None
        while not stop.is_set():
            with self._lock:
                self._check_idle_resync_and_autostow()
            stop.wait(0.2)

    def _check_idle_resync_and_autostow(self) -> None:
        """Resync qpos and optionally auto-stow once both hands have gone idle.

        Whole-body: seed_qpos_from_obs() and follower.park() act on both hands,
        so this fires only when neither is engaged.
        """
        both_disengaged = not self._arms["left"]["engaged"] and not self._arms["right"]["engaged"]
        now = time.monotonic()
        if not both_disengaged:
            self._idle_since = None
            self._auto_stowed_this_idle = False
            return
        if self._idle_since is None:
            self._idle_since = now
        idle_for = now - self._idle_since

        follower = self._stow_follower
        if follower is None:
            from ..follower import get_last_connected_follower
            follower = get_last_connected_follower()
            if follower is not None:
                self._stow_follower = follower
        if follower is None:
            return  # pure-sim / no robot connected — nothing to resync against

        auto_stow_s = self.config.auto_stow_idle_s
        if auto_stow_s is not None and idle_for >= auto_stow_s and not self._auto_stowed_this_idle:
            # Set before the blocking park() so a threaded caller can't double-park.
            self._auto_stowed_this_idle = True
            logger.info("idle %.1fs with both arms disengaged — auto-stowing", idle_for)
            try:
                follower.park()
            except Exception:
                logger.exception("auto-stow park() failed")

        interval = self.config.idle_resync_interval_s
        if interval and now - self._last_idle_resync_t >= interval:
            self._last_idle_resync_t = now
            try:
                self.seed_qpos_from_obs(follower.get_observation())
            except Exception:
                logger.exception("idle resync failed")

    def get_action(self) -> dict[str, float]:
        now = time.perf_counter()
        last = self._last_get_action_t
        self._last_get_action_t = now
        if last is not None:
            dt = max(now - last, 1e-3)
            inst_hz = 1.0 / dt
            self._loop_hz = 0.1 * inst_hz + 0.9 * (self._loop_hz or inst_hz)

        with self._lock:
            xr = self._latest_xr_frame
            last_frame_time = self._last_xr_frame_time
            if xr is None:
                self._check_idle_resync_and_autostow()
                return self._build_action()

            gap_s = time.time() - last_frame_time
            ctrls = xr.get("controllers") or {}
            for hand in ("left", "right"):
                self._update_arm(hand, ctrls.get(hand), gap_s)

            self._check_idle_resync_and_autostow()
            action = self._build_action()

            if self.config.publish_ik_state and self._ws is not None and self._ws_loop is not None:
                self._publish_ik_state_async()

            return action

    # ---------- internals ----------

    def _build_action(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for hand in ("left", "right"):
            arm = self._arms[hand]
            for j in range(ARM_DOFS):
                out[f"{hand}_joint_{j + 1}.pos"] = float(arm["qpos"][j])
            # LeRobot convention: gripper.pos in [0, 1], same polarity as trigger.
            out[f"{hand}_gripper.pos"] = float(np.clip(arm["trigger"], 0.0, 1.0))
        return out

    def _stow_arm(
        self, hand: str, arm: dict, pos: np.ndarray, quat_wxyz: np.ndarray
    ) -> None:
        """Thumbstick STOW: ramp this arm home via the follower, then resync.

        Blocks for the ramp deliberately — the alternative is recording frames
        the operator never commanded. Re-anchors afterwards if still engaged.
        """
        follower = self._stow_follower
        logger.info("%s STOW: ramping home via follower ...", hand)
        try:
            follower.park(hands=[hand])
        except Exception:
            logger.exception("%s stow failed", hand)
            return
        try:
            self.seed_qpos_from_obs(follower.get_observation())
        except Exception:
            logger.exception("%s stow: resync after ramp failed", hand)
            return
        if arm["engaged"]:
            self._anchor_mapper(
                hand, arm, pos, quat_wxyz, "STOW-DONE")
        arm["haptic"] = 0.0
        logger.info("%s STOW done.", hand)

    def _anchor_mapper(
        self, hand: str, arm: dict, pos: np.ndarray, quat_wxyz: np.ndarray, label: str
    ) -> None:
        """Capture the mapper's engage frame at the current robot+controller state.

        label: log tag for the calling path (ENGAGE, RE-ANCHOR, PRECISION, ...).
        """
        ee_pos, ee_quat = arm["solver"].fk(arm["qpos"])
        j4_pos = arm["solver"].j4_anchor_xpos()
        arm["mapper"].engage(pos, quat_wxyz, ee_pos, ee_quat, pivot_armbase=j4_pos)
        arm["engaged"] = True
        logger.debug(
            "%s clutch %s  ee_pos=%s  j4_pos=%s",
            hand, label, ee_pos.round(3),
            j4_pos.round(3) if j4_pos is not None else "—",
        )

    def _update_arm(self, hand: str, ctrl: dict | None, gap_s: float) -> None:
        if ctrl is None:
            return
        arm = self._arms[hand]

        pos_raw = np.asarray(ctrl["position"], dtype=float)
        ox, oy, oz, ow = ctrl["orientation"]  # WebXR sends xyzw
        quat_raw = np.array([ow, ox, oy, oz])

        buttons = ctrl.get("buttons") or []
        # A short button array reads as "never pressed" forever; log the layout
        # once per hand so that failure mode is visible rather than silent.
        if not arm.get("logged_button_layout") and buttons:
            arm["logged_button_layout"] = True
            pressed_idx = [i for i, b in enumerate(buttons) if b.get("p")]
            logger.info(
                "%s controller reports %d buttons (stow triggers on index %d [thumbstick] "
                "or %d [B/Y], grip is index %d; currently pressed: %s)",
                hand, len(buttons), REST_RAMP_BUTTON_INDEX, HANDOFF_BUTTON_INDEX, GRIP_BUTTON_INDEX,
                pressed_idx or "none")
            if len(buttons) <= REST_RAMP_BUTTON_INDEX and len(buttons) <= HANDOFF_BUTTON_INDEX:
                logger.warning(
                    "%s controller exposes only %d buttons — both stow indices (%d, %d) are "
                    "out of range, so stow can never trigger on this hand.",
                    hand, len(buttons), REST_RAMP_BUTTON_INDEX, HANDOFF_BUTTON_INDEX)

        # Read BEFORE the staleness gate: stow needs no fresh pose and must keep
        # working exactly when the connection degrades. B/Y triggers it too.
        thumb_pressed = (
            bool(buttons[REST_RAMP_BUTTON_INDEX]["p"]) if len(buttons) > REST_RAMP_BUTTON_INDEX else False
        )
        handoff_pressed = (
            bool(buttons[HANDOFF_BUTTON_INDEX]["p"]) if len(buttons) > HANDOFF_BUTTON_INDEX else False
        )
        rest_btn = thumb_pressed or handoff_pressed
        if rest_btn and not arm["last_rest_button"] and not arm["ramp_active"]:
            # Always announce: silence here means the index is wrong, which is a
            # different problem from "stow ran but the arm didn't move".
            logger.info("%s %s pressed (rising edge)", hand, "thumbstick" if thumb_pressed else "B/Y")
            # Resolve lazily: connect() order isn't enforced, and trusting it would
            # silently degrade stow to the rest_qpos ramp.
            follower = self._stow_follower
            if follower is None:
                from ..follower import get_last_connected_follower
                follower = get_last_connected_follower()
                if follower is not None:
                    self._stow_follower = follower
                    logger.info("stow: bound to %s on first use", follower)
            arm["last_rest_button"] = rest_btn
            if follower is not None:
                # The follower's joint-space ramp targets the powered-up pose, not
                # the generic rest_qpos, and keeps IK out of the loop entirely.
                self._stow_arm(hand, arm, pos_raw, quat_raw)
                return
            logger.warning(
                "%s thumbstick: no follower registered — falling back to the internal "
                "rest_qpos ramp (target %s). On hardware this means stow will move the "
                "arm to that generic pose, not the pose it powered up in.",
                hand, np.round(self.config.rest_qpos_left if hand == "left"
                               else self.config.rest_qpos_right, 3).tolist())
            target = self.config.rest_qpos_left if hand == "left" else self.config.rest_qpos_right
            arm["ramp_start_q"] = arm["qpos"][:ARM_DOFS].copy()
            arm["ramp_target_q"] = np.asarray(target, dtype=float)
            arm["ramp_start_t"] = time.perf_counter()
            arm["ramp_active"] = True
            logger.debug(
                "%s rest-ramp START (duration=%.2fs, target=%s, engaged=%s)",
                hand,
                self.config.rest_ramp_duration_s,
                arm["ramp_target_q"].round(3),
                arm["engaged"],
            )
        else:
            arm["last_rest_button"] = rest_btn

        # Stale pose: skip engage/disengage/IK and flag a re-anchor, so the
        # engage-delta restarts at zero and there is no catch-up motion.
        if gap_s > XR_FRAME_STALE_TIMEOUT_S:
            if arm["engaged"] and not arm["needs_reanchor"]:
                arm["needs_reanchor"] = True
                logger.warning("%s xr_frame stale (%.2fs gap) — pausing", hand, gap_s)
            arm["haptic"] *= 0.6  # decay so vibration doesn't linger
            return

        # nlerp with a hemisphere check — without it the lerp can drift to the
        # antipodal point and flip. First tick copies raw instead of lerping.
        alpha = float(self.config.pose_filter_alpha)
        if arm["pos_filt"] is None or arm["quat_filt"] is None:
            arm["pos_filt"] = pos_raw.copy()
            arm["quat_filt"] = quat_raw.copy()
        else:
            arm["pos_filt"] = (1.0 - alpha) * arm["pos_filt"] + alpha * pos_raw
            q_in = quat_raw if np.dot(arm["quat_filt"], quat_raw) >= 0.0 else -quat_raw
            qf = (1.0 - alpha) * arm["quat_filt"] + alpha * q_in
            arm["quat_filt"] = qf / np.linalg.norm(qf)
        pos = arm["pos_filt"]
        quat_wxyz = arm["quat_filt"]

        grip = bool(buttons[GRIP_BUTTON_INDEX]["p"]) if len(buttons) > GRIP_BUTTON_INDEX else False
        trigger = float(buttons[TRIGGER_BUTTON_INDEX]["v"]) if len(buttons) > TRIGGER_BUTTON_INDEX else 0.0
        precision = (
            bool(buttons[PRECISION_BUTTON_INDEX]["p"]) if len(buttons) > PRECISION_BUTTON_INDEX else False
        )
        # Precision scales the mapper gains down; re-anchor on either edge so the
        # accumulated delta is not reinterpreted under the new scale.
        if precision != arm["last_precision"] and arm["engaged"]:
            self._anchor_mapper(
                hand, arm, pos, quat_wxyz, "PRECISION" if precision else "FULL-SCALE"
            )
        arm["last_precision"] = precision
        scale_factor = self.config.precision_factor if precision else 1.0
        arm["mapper"].scale = self.config.scale_translation * scale_factor
        arm["mapper"].scale_rotation = self.config.scale_rotation * scale_factor
        arm["mapper"].rot_reach_limit = self.config.rot_reach_limit
        arm["mapper"].pos_reach_limit = self.config.pos_reach_limit

        # Re-anchor after a stale window if the grip is still held; a release
        # during the stall falls through to the normal disengage edge below.
        if arm["needs_reanchor"] and arm["engaged"] and grip:
            self._anchor_mapper(
                hand, arm, pos, quat_wxyz, "RE-ANCHOR")
        arm["needs_reanchor"] = False

        # Edge-detect clutch.
        if grip and not arm["last_grip"]:
            self._anchor_mapper(
                hand, arm, pos, quat_wxyz, "ENGAGE")
        elif not grip and arm["last_grip"]:
            arm["mapper"].disengage()
            arm["engaged"] = False
            # Dropped on release so the next engage has no lerp-lag from old state.
            arm["pos_filt"] = None
            arm["quat_filt"] = None
            logger.debug("%s clutch RELEASE", hand)
        arm["last_grip"] = grip

        # Engaged only: otherwise stray trigger pressure would move the gripper
        # outside an active intervention.
        if arm["engaged"]:
            arm["trigger"] = trigger
            grip_qpos = GRIPPER_QPOS_OPEN + trigger * (GRIPPER_QPOS_CLOSED - GRIPPER_QPOS_OPEN)
            arm["qpos"][6] = grip_qpos
            arm["qpos"][7] = grip_qpos

        # The ramp owns qpos[:6] while active; IK is skipped so the mapper delta
        # doesn't fight the interpolation. The gripper still tracks the trigger.
        if arm["ramp_active"]:
            duration = max(1e-3, float(self.config.rest_ramp_duration_s))
            elapsed = time.perf_counter() - arm["ramp_start_t"]
            t = min(1.0, elapsed / duration)
            arm["qpos"][:ARM_DOFS] = arm["ramp_start_q"] + t * (arm["ramp_target_q"] - arm["ramp_start_q"])
            if t >= 1.0:
                arm["ramp_active"] = False
                if arm["engaged"]:
                    # Re-anchor at the rest EE so the first IK tick doesn't jump.
                    self._anchor_mapper(
                hand, arm, pos, quat_wxyz, "RAMP-DONE")
                else:
                    logger.debug("%s rest-ramp DONE (disengaged)", hand)
            arm["haptic"] *= 0.6  # decay haptic during the ramp
            return

        # FK at the last commanded qpos; feeds the mapper's reach limits.
        ee_pos_now, ee_quat_now = arm["solver"].fk(arm["qpos"])
        out = arm["mapper"].target(pos, quat_wxyz, ee_pos_now, ee_quat_now)
        if out is not None:
            tgt_pos, tgt_quat = out
            arm["qpos"][:ARM_DOFS] = arm["solver"].solve(tgt_pos, tgt_quat, arm["qpos"])

            # Max of four 0..1 cues: limit pressure, reach error, singularity and
            # gimbal proximity — near gimbal the wrist goes sluggish, not limited.
            pressure = float(getattr(arm["solver"], "last_limit_pressure", 0.0))
            pos_err = float(getattr(arm["solver"], "last_pos_err_norm", 0.0))
            singular = float(getattr(arm["solver"], "last_singularity_proximity", 0.0))
            gimbal = float(getattr(arm["solver"], "last_wrist_gimbal_proximity", 0.0))
            i_limit = min(1.0, max(0.0, (pressure - 0.05) / 0.25))
            i_reach = min(1.0, max(0.0, (pos_err - 0.03) / 0.10))
            i_singular = max(0.0, (singular - 0.95) / 0.2)
            i_gimbal = min(1.0, max(0.0, (gimbal - 0.5) / 0.5))
            raw_intensity = max(i_limit, i_reach, i_singular, i_gimbal)
            arm["haptic"] = 0.6 * arm["haptic"] + 0.4 * raw_intensity
        else:
            arm["haptic"] *= 0.6  # decay so vibration doesn't linger after release

    def _publish_ik_state_async(self) -> None:
        """Schedule an ik_state send: called on the LeRobot thread, run on the WS thread."""
        payload = {
            "type": "ik_state",
            "left_qpos": [float(v) for v in self._arms["left"]["qpos"][:NQ]],
            "right_qpos": [float(v) for v in self._arms["right"]["qpos"][:NQ]],
            "left_engaged": bool(self._arms["left"]["engaged"]),
            "right_engaged": bool(self._arms["right"]["engaged"]),
            # EMA'd max of the IK trouble signals; vibrates the matching controller.
            "left_haptic": float(self._arms["left"]["haptic"]),
            "right_haptic": float(self._arms["right"]["haptic"]),
            # From `send_feedback`'s gripper torque; mixed with `*_haptic` client-side.
            "left_force_haptic": float(self._arms["left"]["force_haptic"]),
            "right_force_haptic": float(self._arms["right"]["force_haptic"]),
            # Compat shim: the Quest UI still reads single-arm `qpos`.
            "qpos": [float(v) for v in self._arms["right"]["qpos"][:NQ]],
            "engaged": bool(self._arms["right"]["engaged"]),
            # Lets passive listeners pick one stream when several teleops share a relay.
            "teleop_id": str(self.config.id) if self.config.id is not None else None,
            "server_time": time.time(),
            # 0.0 until the second call; the UI treats <1 Hz as unknown.
            "loop_hz": float(self._loop_hz or 0.0),
        }
        text = json.dumps(payload)

        async def _send():
            try:
                if self._ws is not None:
                    await self._ws.send(text)
            except Exception:
                pass

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._ws_loop)
        except Exception:
            pass

    # ---------- Live config (from the web UI) ----------

    # Web-tunable knobs and their hard safety bounds; out-of-range values are
    # clamped with a warning so a runaway slider can't reach unsafe gains.
    _LIVE_CONFIG_BOUNDS: dict[str, tuple[float, float]] = {
        "scale_translation": (0.1, 10.0),
        "scale_rotation": (0.1, 10.0),
        "pose_filter_alpha": (0.05, 1.0),
        "precision_factor": (0.05, 1.0),
        # Gripper-torque deadband (Nm): too low buzzes at idle, too high hides
        # grasps. Upper bound matches the follower's max_gripper_torque.
        "force_haptic_threshold_nm_left": (0.0, 1.0),
        "force_haptic_threshold_nm_right": (0.0, 1.0),
        # Per-group Δq cap (rad/tick): joints 1-3 and 4-6. The side-effect below
        # rebuilds the 6-vector and pushes it into both solvers.
        "max_dq_per_joint_scalar_pos": (0.005, 0.50),
        "max_dq_per_joint_scalar_rot": (0.005, 0.50),
    }

    # On/off counterpart to `_LIVE_CONFIG_BOUNDS`; set directly, no clamping.
    _LIVE_CONFIG_BOOLS: tuple[str, ...] = ("force_haptic_enabled",)

    def _apply_config_update(self, cfg: dict) -> None:
        """Apply a `config_update` payload from the web UI.

        Mutates `self.config` in place under the lock; `_update_arm` re-reads it
        every tick, so the new value lands on the next solve.

        cfg: {field: value} for keys in _LIVE_CONFIG_BOUNDS / _LIVE_CONFIG_BOOLS.
        """
        if not isinstance(cfg, dict):
            return
        applied: dict[str, float | bool] = {}
        with self._lock:
            for key, (lo, hi) in self._LIVE_CONFIG_BOUNDS.items():
                if key not in cfg:
                    continue
                try:
                    v = float(cfg[key])
                except (TypeError, ValueError):
                    logger.warning("config_update: %s=%r not a number; ignoring", key, cfg[key])
                    continue
                if not (lo <= v <= hi):
                    logger.warning(
                        "config_update: %s=%.3f out of range [%.2f, %.2f]; clamping", key, v, lo, hi
                    )
                    v = max(lo, min(hi, v))
                setattr(self.config, key, v)
                applied[key] = v
            # Once after the loop, so a payload touching both groups writes once.
            if (
                "max_dq_per_joint_scalar_pos" in applied
                or "max_dq_per_joint_scalar_rot" in applied
            ):
                pos = float(self.config.max_dq_per_joint_scalar_pos)
                rot = float(self.config.max_dq_per_joint_scalar_rot)
                arr = [pos] * 3 + [rot] * 3
                self.config.max_dq_per_joint = arr
                for hand in ("left", "right"):
                    solver = self._arms[hand].get("solver")
                    if solver is not None:
                        solver.max_dq_per_joint = np.asarray(arr, dtype=float).copy()
            for key in self._LIVE_CONFIG_BOOLS:
                if key not in cfg:
                    continue
                v_bool = bool(cfg[key])
                setattr(self.config, key, v_bool)
                applied[key] = v_bool
        if applied:
            logger.info(
                "config_update applied: %s",
                ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in applied.items()),
            )

    # ---------- Haptic-threshold calibration (web UI button) ----------

    # Margin over the observed idle peak: a short sample misses the session tail.
    # 0.20 silences idle buzz without eating the contact range (>0.5 Nm here).
    _HAPTIC_CALIB_MARGIN_NM: float = 0.20
    # Ceiling on auto-written thresholds, so a faulty gripper can't silence the
    # haptic entirely. Matches the _LIVE_CONFIG_BOUNDS upper bound.
    _HAPTIC_CALIB_MAX_THRESHOLD_NM: float = 1.0
    # Minimum (max − threshold) span kept after calibration, so the deadband
    # always has room to ramp instead of saturating every grasp to max.
    _HAPTIC_CALIB_MIN_SPAN_NM: float = 0.50
    # Peak above this hints at a hardware fault; calibration warns rather than
    # letting the operator trust the result.
    _HAPTIC_CALIB_SUSPICIOUS_PEAK_NM: float = 0.50

    def _start_haptic_calibration(self, duration_s: float = 5.0) -> None:
        """Begin a per-arm idle-torque sampling window.

        duration_s: window length in seconds, clamped to [0.5, 10.0].
        """
        duration_s = max(0.5, min(10.0, float(duration_s)))
        with self._lock:
            if self._haptic_calib is not None:
                logger.info("haptic_calibrate: already running; ignoring duplicate start")
                return
            self._haptic_calib = {
                "start_t": time.time(),
                "duration_s": duration_s,
                "peak": {"left": 0.0, "right": 0.0},
            }
        logger.info("haptic_calibrate: starting %.1fs idle-torque sampling", duration_s)

    def _finalize_haptic_calibration(self) -> None:
        """Close the calibration window, write thresholds, emit the result message.

        Must be called with `self._lock` held.
        """
        calib = self._haptic_calib
        if calib is None:
            return
        margin = self._HAPTIC_CALIB_MARGIN_NM
        threshold_cap = self._HAPTIC_CALIB_MAX_THRESHOLD_NM
        peaks = calib["peak"]
        new = {hand: min(threshold_cap, peaks[hand] + margin) for hand in ("left", "right")}
        self.config.force_haptic_threshold_nm_left = new["left"]
        self.config.force_haptic_threshold_nm_right = new["right"]

        # Keep the span ≥ MIN_SPAN_NM so grasps stay unsaturated. Raise only.
        required_max = max(new["left"], new["right"]) + self._HAPTIC_CALIB_MIN_SPAN_NM
        old_max = float(self.config.force_haptic_max_nm)
        new_max = max(old_max, required_max)
        if new_max != old_max:
            self.config.force_haptic_max_nm = new_max

        # Still written, so the operator isn't stuck with constant buzz, but flagged.
        suspicious = []
        for hand in ("left", "right"):
            if peaks[hand] > self._HAPTIC_CALIB_SUSPICIOUS_PEAK_NM:
                suspicious.append(f"{hand} peak={peaks[hand]:.3f} Nm")
        if suspicious:
            logger.warning(
                "haptic_calibrate: suspicious idle peak(s) — %s. "
                "Expected <%.2f Nm at rest with empty jaws; check the gripper "
                "isn't jammed, the cable isn't fouled on the frame, and the "
                "command isn't holding closed against a stop.",
                ", ".join(suspicious), self._HAPTIC_CALIB_SUSPICIOUS_PEAK_NM,
            )

        self._haptic_calib = None
        logger.info(
            "haptic_calibrate done: peak L=%.3f R=%.3f → threshold L=%.3f R=%.3f Nm "
            "(margin=%.2f, force_haptic_max_nm=%.2f)",
            peaks["left"], peaks["right"], new["left"], new["right"], margin, new_max,
        )
        # Broadcast back to the web client so it can display the new values.
        # Fire-and-forget on the WS loop so we don't block under the lock.
        payload = {
            "type": "haptic_calibrate_result",
            "left_peak_nm": float(peaks["left"]),
            "right_peak_nm": float(peaks["right"]),
            "left_threshold_nm": float(new["left"]),
            "right_threshold_nm": float(new["right"]),
            "margin_nm": float(margin),
            "max_nm": float(new_max),
            "suspicious": suspicious,
        }
        text = json.dumps(payload)

        async def _send() -> None:
            try:
                if self._ws is not None:
                    await self._ws.send(text)
            except Exception:
                pass

        if self._ws_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(_send(), self._ws_loop)
            except Exception:
                pass

    # ---------- WS thread ----------

    def _ws_thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._ws_loop = loop
        try:
            loop.run_until_complete(self._ws_runner())
        finally:
            loop.close()
            self._ws_loop = None

    def _ssl_context_for(self, url: str):
        """Build an SSL context for `wss://` URLs, or None for `ws://`.

        Validation is skipped for a local relay, whose self-signed cert would
        otherwise be rejected; remote targets use the default trust store.

        url: the websocket URL being connected to.
        """
        if not url.startswith("wss://"):
            return None
        import ssl

        ctx = ssl.create_default_context()
        if "://localhost" in url or "://127.0.0.1" in url:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def _ws_runner(self) -> None:
        backoff = 1.0
        ssl_ctx = self._ssl_context_for(self.config.ws_url)
        while not self._ws_stop.is_set():
            try:
                async with websockets.connect(self.config.ws_url, ssl=ssl_ctx) as ws:
                    self._ws = ws
                    self._ws_connected.set()
                    backoff = 1.0
                    # Ask the page to re-broadcast its sliders so the session doesn't
                    # start on stale dataclass defaults. No-op if no page is connected.
                    try:
                        await ws.send(json.dumps({"type": "request_settings"}))
                    except Exception as e:
                        logger.debug("request_settings send failed: %s", e)
                    async for raw in ws:
                        if self._ws_stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        mtype = msg.get("type")
                        if mtype == "xr_frame":
                            # Extracted here so the DAgger listener stays live even
                            # when get_action(), and thus the IK pipeline, isn't.
                            btn_snapshot = {
                                "left": {"grip": False, "handoff": False},
                                "right": {"grip": False, "handoff": False},
                            }
                            ctrls = msg.get("controllers") or {}
                            for hand in ("left", "right"):
                                ctrl = ctrls.get(hand) or {}
                                buttons = ctrl.get("buttons") or []
                                if len(buttons) > GRIP_BUTTON_INDEX:
                                    btn_snapshot[hand]["grip"] = bool(buttons[GRIP_BUTTON_INDEX].get("p"))
                                if len(buttons) > HANDOFF_BUTTON_INDEX:
                                    btn_snapshot[hand]["handoff"] = bool(
                                        buttons[HANDOFF_BUTTON_INDEX].get("p")
                                    )
                            with self._lock:
                                self._latest_xr_frame = msg
                                self._last_xr_frame_time = time.time()
                                self._buttons = btn_snapshot
                        elif mtype == "config_update":
                            self._apply_config_update(msg.get("config") or {})
                        elif mtype == "haptic_calibrate":
                            self._start_haptic_calibration(
                                float(msg.get("duration_s") or 5.0)
                            )
                        # else: ignore (our own echoed ik_state, pings, etc.)
            except Exception as e:
                logger.warning(
                    "BiQuestTeleoperator WS error (%s); reconnecting in %.1fs", type(e).__name__, backoff
                )
            finally:
                self._ws = None
                self._ws_connected.clear()
            if self._ws_stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 10.0)
