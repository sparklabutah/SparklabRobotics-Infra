"""Bimanual VR teleoperation via WebXR/Quest.

Reads `xr_frame` from the relay's /ws; per arm keeps a ClutchPoseMapper and a
DecoupledIKSolver. Per tick `_update_arm` runs: staleness gate -> EMA pose
filter -> precision button -> clutch edges -> IK step -> haptic mix.

get_action() returns {left,right}_joint_{1..6}.pos (rad) and
{left,right}_gripper.pos (0..1), matching YamUltraFollower.action_features.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np

try:
    from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "lerobot is required to use BiQuestTeleoperator. "
        "Run from a project environment with lerobot installed."
    ) from e

from ....core.pose_mapping import ClutchPoseMapper
from ....quest import XRFrameClient, buttons as xr_buttons
from ..model.kinematics import DEFAULT_Q_REST
from ..ik.decoupled_ik import DecoupledIKSolver
from .haptics import ForceHaptics

logger = logging.getLogger(__name__)


# Default rotation taking Quest `local-floor` world axes into the arm base
DEFAULT_R_CALIB = [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

ARM_DOFS = 6
NQ = 8  # arm (6) + 2 gripper-finger sliders

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



class ArmState(TypedDict):
    solver: DecoupledIKSolver
    mapper: ClutchPoseMapper
    qpos: np.ndarray
    last_grip: bool
    last_precision: bool
    trigger: float
    engaged: bool
    haptic: float
    force_haptic: float
    needs_reanchor: bool
    pos_filt: np.ndarray | None
    quat_filt: np.ndarray | None
    last_rest_button: bool
    ramp_active: bool
    ramp_start_q: np.ndarray
    ramp_target_q: np.ndarray
    ramp_start_t: float
    logged_button_layout: bool


class BiQuestTeleoperator(Teleoperator):
    config_class = BiQuestTeleoperatorConfig
    name = "bi_quest_teleop"

    def __init__(
        self,
        config: BiQuestTeleoperatorConfig,
        hands: tuple[str, ...] = ("left", "right"),
    ) -> None:
        super().__init__(config)
        self.config = config

        self._r_calib = np.asarray(config.r_calib, dtype=float)
        if self._r_calib.shape != (3, 3):
            raise ValueError(
                f"r_calib must be a 3x3 rotation matrix, got shape {self._r_calib.shape}"
            )

        if not hands or any(hand not in ("left", "right") for hand in hands):
            raise ValueError(f"hands must contain left and/or right, got {hands!r}")
        self._hands = tuple(dict.fromkeys(hands))
        self._lock = threading.RLock()
        self._latest_xr_frame: dict | None = None
        self._last_xr_frame_time: float = 0.0

        rest_qpos = {
            "left": np.asarray(config.rest_qpos_left, dtype=float),
            "right": np.asarray(config.rest_qpos_right, dtype=float),
        }
        for hand, q in rest_qpos.items():
            if q.shape != (ARM_DOFS,):
                raise ValueError(f"rest_qpos_{hand} must have {ARM_DOFS} values, got shape {q.shape}")

        self._arms: dict[str, ArmState] = {}
        for hand in self._hands:
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
                "needs_reanchor": False,  # set during stale → re-anchor on recovery
                "pos_filt": None,  # EMA-smoothed controller position
                "quat_filt": None,  # EMA-smoothed controller orientation (wxyz)
                "last_rest_button": False,
                "ramp_active": False,
                "ramp_start_q": np.zeros(ARM_DOFS),
                "ramp_target_q": q_rest.copy(),
                "ramp_start_t": 0.0,
                "logged_button_layout": False,
            }
        self._buttons: dict[str, dict[str, bool]] = {
            hand: {"grip": False, "handoff": False} for hand in self._hands
        }
        self._relay = XRFrameClient(
            config.ws_url, name="quest-teleop-relay", on_message=self._on_relay_message
        )

        self._force_haptics = ForceHaptics(config, self._hands)

        # EMA-smoothed get_action() rate, published in ik_state so the web UI can
        # show the Δq-cap slider in honest rad/s.
        self._last_get_action_t: float | None = None
        self._loop_hz: float | None = None

    # ---------- Teleoperator interface ----------

    @property
    def action_features(self) -> dict[str, type]:
        feats: dict[str, type] = {}
        for hand in self._hands:
            for j in range(1, ARM_DOFS + 1):
                feats[f"{hand}_joint_{j}.pos"] = float
            feats[f"{hand}_gripper.pos"] = float
        return feats

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._relay.is_connected

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
        self._relay.connect(timeout_s=self.config.connect_timeout_s)
        logger.info("BiQuestTeleoperator connected to %s", self.config.ws_url)
        self._relay.send({"type": "request_settings"})

    def disconnect(self) -> None:
        self._relay.disconnect()
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
        if self.config.publish_ik_state:
            self._publish_ik_state_async()

    def _apply_torque_feedback(self, torques: dict) -> None:
        """Update per-arm force feedback and publish completed calibration."""
        with self._lock:
            intensities, result = self._force_haptics.update(torques)
            for hand, intensity in intensities.items():
                self._arms[hand]["force_haptic"] = intensity
        if result is not None:
            if result["suspicious"]:
                logger.warning(
                    "haptic calibration found high idle torque: %s",
                    ", ".join(result["suspicious"]),
                )
            self._relay.send(result)

    # ---------- Intervention (handoff) hooks ----------
    # Level, not edge. Reads `self._buttons`, which the WS reader keeps fresh.

    def is_engaged(self) -> bool:
        with self._lock:
            return any(self._buttons[hand]["grip"] for hand in self._hands)

    def is_handoff_pressed(self) -> bool:
        with self._lock:
            return any(self._buttons[hand]["handoff"] for hand in self._hands)

    def is_pause_pressed(self) -> bool:
        """Right-hand B — pause/resume the policy. Level only."""
        with self._lock:
            return self._buttons.get("right", {}).get("handoff", False)

    def is_reverse_pressed(self) -> bool:
        """Left-hand Y — reverse, e.g. replay buffered actions backward. Level only."""
        with self._lock:
            return self._buttons.get("left", {}).get("handoff", False)

    def seed_qpos_from_obs(self, obs: dict[str, float]) -> None:
        """Reset qpos / gripper / engagement to match a `.pos`-keyed dict.

        Also force-disengages, so a grip released while nothing was calling
        get_action() cannot leave a stale mapper anchor behind.

        obs: robot observation or commanded action. Prefer the command at
            handoff — it keeps a gripper squeezing an object closed rather than
            recording its blocked-open measurement. Missing keys are skipped.
        """
        with self._lock:
            for hand in self._hands:
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
                return self._build_action()

            gap_s = time.time() - last_frame_time
            ctrls = xr.get("controllers") or {}
            for hand in self._hands:
                self._update_arm(hand, ctrls.get(hand), gap_s)

            action = self._build_action()

            if self.config.publish_ik_state:
                self._publish_ik_state_async()

            return action

    # ---------- internals ----------

    def _build_action(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for hand in self._hands:
            arm = self._arms[hand]
            for j in range(ARM_DOFS):
                out[f"{hand}_joint_{j + 1}.pos"] = float(arm["qpos"][j])
            # LeRobot convention: gripper.pos in [0, 1], same polarity as trigger.
            out[f"{hand}_gripper.pos"] = float(np.clip(arm["trigger"], 0.0, 1.0))
        return out

    def _stow_arm(self, hand: str, arm: ArmState) -> None:
        """Thumbstick STOW: ramp this arm home without blocking the loop.

        The ramp is advanced a step per tick by _update_arm, so the control
        loop keeps running: the other hand stays live and a second stow press
        is seen immediately. Blocking here (the old follower.park() call) is
        what made two stows run back to back for the full duration each, and
        why the second press was not even detected until the first finished.

        Targets rest_qpos, which the bridge sets to the pose the arms powered
        up in, so a stowed arm and the bridge's idea of home agree. park()
        ramps to zero instead and stays what it is: the safe pose for cutting
        torque on disconnect.
        """
        target = (
            self.config.rest_qpos_left if hand == "left" else self.config.rest_qpos_right
        )
        arm["ramp_start_q"] = arm["qpos"][:ARM_DOFS].copy()
        arm["ramp_target_q"] = np.asarray(target, dtype=float)
        arm["ramp_start_t"] = time.perf_counter()
        arm["ramp_active"] = True
        arm["haptic"] = 0.0
        logger.info(
            "%s STOW: ramping home over %.1fs (non-blocking, target=%s)",
            hand, self.config.rest_ramp_duration_s,
            arm["ramp_target_q"].round(3).tolist(),
        )

    def _anchor_mapper(
        self, hand: str, arm: dict, pos: np.ndarray, quat_wxyz: np.ndarray, label: str
    ) -> None:
        """Capture the mapper's engage frame at the current robot+controller state.

        label: log tag for the calling path (ENGAGE, RE-ANCHOR, PRECISION, ...).
        """
        # Anchors at the last COMMANDED qpos, never the measurement: measured
        # lags command by the tracking error. See DESIGN.md#teleoperation.
        ee_pos, ee_quat = arm["solver"].fk(arm["qpos"])
        j4_pos = arm["solver"].j4_anchor_xpos()
        arm["mapper"].engage(pos, quat_wxyz, ee_pos, ee_quat, pivot_armbase=j4_pos)
        arm["engaged"] = True
        logger.info(
            "%s clutch %s  ee_pos=%s  j4_pos=%s",
            hand, label, ee_pos.round(3),
            j4_pos.round(3) if j4_pos is not None else "—",
        )

    def _advance_ramp(
        self,
        hand: str,
        arm: ArmState,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        pose_is_fresh: bool,
    ) -> bool:
        """Advance a stow trajectory even when the XR stream is stale."""
        if not arm["ramp_active"]:
            return False
        duration = max(1e-3, float(self.config.rest_ramp_duration_s))
        elapsed = time.perf_counter() - arm["ramp_start_t"]
        t = min(1.0, elapsed / duration)
        arm["qpos"][:ARM_DOFS] = arm["ramp_start_q"] + t * (
            arm["ramp_target_q"] - arm["ramp_start_q"]
        )
        if t >= 1.0:
            arm["ramp_active"] = False
            if arm["engaged"] and pose_is_fresh:
                self._anchor_mapper(hand, arm, pos, quat_wxyz, "RAMP-DONE")
            elif arm["engaged"]:
                arm["needs_reanchor"] = True
        arm["haptic"] *= 0.6
        return True

    def _update_arm(self, hand: str, ctrl: dict | None, gap_s: float) -> None:
        if ctrl is None:
            return
        arm = self._arms[hand]

        pos_raw = np.asarray(ctrl["position"], dtype=float)
        ox, oy, oz, ow = ctrl["orientation"]  # WebXR sends xyzw
        quat_raw = np.array([ow, ox, oy, oz])

        buttons = ctrl.get("buttons") or []
        if not arm.get("logged_button_layout") and buttons:
            arm["logged_button_layout"] = True
            pressed_idx = [i for i, b in enumerate(buttons) if b.get("p")]
            logger.info(
                "%s controller reports %d buttons (stow triggers on index %d [thumbstick] "
                "or %d [B/Y], grip is index %d; currently pressed: %s)",
                hand, len(buttons), xr_buttons.THUMBSTICK, xr_buttons.B_Y, xr_buttons.GRIP,
                pressed_idx or "none")
            if len(buttons) <= xr_buttons.THUMBSTICK:
                if len(buttons) > xr_buttons.B_Y:
                    logger.warning(
                        "%s controller exposes only %d buttons — no thumbstick at index %d, "
                        "so stow falls back to B/Y on this hand and will also toggle "
                        "the bridge's ARMED state.",
                        hand, len(buttons), xr_buttons.THUMBSTICK)
                else:
                    logger.warning(
                        "%s controller exposes only %d buttons — neither stow index (%d, %d) "
                        "is in range, so stow can never trigger on this hand.",
                        hand, len(buttons), xr_buttons.THUMBSTICK, xr_buttons.B_Y)

        thumb_pressed = xr_buttons.pressed(buttons, xr_buttons.THUMBSTICK)
        handoff_pressed = xr_buttons.pressed(buttons, xr_buttons.B_Y)

        # B/Y belongs to the bridge: it toggles ARMED, and is_handoff_pressed()
        # ORs the two hands, so the bridge cannot tell which one pressed. Wiring
        # stow to B/Y as well meant one press did both jobs - the first hand
        # armed and stowed, the second hand's press toggled the bridge straight
        # back to DISARMED, and only the first arm ever moved. Stow is the
        # thumbstick; B/Y is only a fallback for a controller that has none.
        has_thumbstick = len(buttons) > xr_buttons.THUMBSTICK
        rest_btn = thumb_pressed if has_thumbstick else handoff_pressed
        if rest_btn and not arm["last_rest_button"] and not arm["ramp_active"]:
            # Always announce: silence here means the index is wrong, which is a
            # different problem from "stow ran but the arm didn't move".
            logger.info(
                "%s %s pressed (rising edge) — stowing this arm",
                hand, "thumbstick" if thumb_pressed else "B/Y (no thumbstick on this controller)",
            )
            arm["last_rest_button"] = rest_btn
            # One stow path now. It ramps to rest_qpos, which the bridge sets to
            # the powered-up pose, and it does not block: the old follower.park()
            # branch held the control loop for the whole ramp, so two arms could
            # never stow at the same time and the second press went unseen.
            self._stow_arm(hand, arm)
            return
        else:
            arm["last_rest_button"] = rest_btn

        pose_is_fresh = gap_s <= xr_buttons.XR_FRAME_STALE_TIMEOUT_S
        if self._advance_ramp(hand, arm, pos_raw, quat_raw, pose_is_fresh):
            return

        # Stale pose: skip engage/disengage/IK and flag a re-anchor, so the
        # engage-delta restarts at zero and there is no catch-up motion.
        if gap_s > xr_buttons.XR_FRAME_STALE_TIMEOUT_S:
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

        grip = xr_buttons.pressed(buttons, xr_buttons.GRIP)
        trigger = xr_buttons.value(buttons, xr_buttons.TRIGGER)
        precision = xr_buttons.pressed(buttons, xr_buttons.A_X)
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
            self._anchor_mapper(hand, arm, pos, quat_wxyz, "RE-ANCHOR")
        arm["needs_reanchor"] = False

        # Edge-detect clutch.
        if grip and not arm["last_grip"]:
            self._anchor_mapper(hand, arm, pos, quat_wxyz, "ENGAGE")
        elif not grip and arm["last_grip"]:
            arm["mapper"].disengage()
            arm["engaged"] = False
            # Dropped on release so the next engage has no lerp-lag from old state.
            arm["pos_filt"] = None
            arm["quat_filt"] = None
            logger.info("%s clutch RELEASE", hand)
        arm["last_grip"] = grip

        # Engaged only: otherwise stray trigger pressure would move the gripper
        # outside an active intervention.
        if arm["engaged"]:
            arm["trigger"] = trigger
            grip_qpos = GRIPPER_QPOS_OPEN + trigger * (GRIPPER_QPOS_CLOSED - GRIPPER_QPOS_OPEN)
            arm["qpos"][6] = grip_qpos
            arm["qpos"][7] = grip_qpos

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
        """Queue an ``ik_state`` message on the shared relay client."""
        def state(hand: str) -> tuple[list[float], bool, float, float]:
            arm = self._arms.get(hand)
            if arm is None:
                return [0.0] * NQ, False, 0.0, 0.0
            return (
                [float(v) for v in arm["qpos"][:NQ]],
                bool(arm["engaged"]),
                float(arm["haptic"]),
                float(arm["force_haptic"]),
            )

        left_q, left_engaged, left_haptic, left_force = state("left")
        right_q, right_engaged, right_haptic, right_force = state("right")
        payload = {
            "type": "ik_state",
            "left_qpos": left_q,
            "right_qpos": right_q,
            "left_engaged": left_engaged,
            "right_engaged": right_engaged,
            "left_haptic": left_haptic,
            "right_haptic": right_haptic,
            "left_force_haptic": left_force,
            "right_force_haptic": right_force,
            "teleop_id": str(self.config.id) if self.config.id is not None else None,
            "server_time": time.time(),
            "loop_hz": float(self._loop_hz or 0.0),
        }
        self._relay.send(payload)

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
                for hand in self._hands:
                    self._arms[hand]["solver"].max_dq_per_joint = np.asarray(
                        arr, dtype=float
                    ).copy()
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

    def _start_haptic_calibration(self, duration_s: float) -> None:
        with self._lock:
            started = self._force_haptics.start_calibration(duration_s)
        if not started:
            logger.info("haptic calibration is already running")

    def _on_relay_message(self, msg: dict) -> None:
        """Dispatch non-transport relay messages on the relay thread."""
        mtype = msg.get("type")
        if mtype == "xr_frame":
            snapshot: dict[str, dict[str, bool]] = {}
            controllers = msg.get("controllers") or {}
            for hand in self._hands:
                raw_buttons = (controllers.get(hand) or {}).get("buttons") or []
                snapshot[hand] = {
                    "grip": xr_buttons.pressed(raw_buttons, xr_buttons.GRIP),
                    "handoff": xr_buttons.pressed(raw_buttons, xr_buttons.B_Y),
                }
            with self._lock:
                self._latest_xr_frame = msg
                self._last_xr_frame_time = time.time()
                self._buttons = snapshot
        elif mtype == "config_update":
            self._apply_config_update(msg.get("config") or {})
        elif mtype == "haptic_calibrate":
            self._start_haptic_calibration(float(msg.get("duration_s") or 5.0))
