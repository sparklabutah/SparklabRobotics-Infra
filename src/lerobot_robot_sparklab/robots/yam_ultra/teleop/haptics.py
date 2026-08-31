from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _HandForceState:
    previous_position: float | None = None
    previous_time: float | None = None
    velocity: float = 0.0
    intensity: float = 0.0


class ForceHaptics:
    """Gripper torque filtering and idle-threshold calibration."""

    CALIBRATION_MARGIN_NM = 0.20
    MIN_CALIBRATED_SPAN_NM = 0.50
    SUSPICIOUS_IDLE_PEAK_NM = 0.50

    def __init__(self, config, hands: tuple[str, ...]) -> None:
        self.config = config
        self.hands = hands
        self._states = {hand: _HandForceState() for hand in hands}
        self._calibration: dict | None = None

    def start_calibration(self, duration_s: float) -> bool:
        if self._calibration is not None:
            return False
        self._calibration = {
            "start": time.time(),
            "duration": max(0.5, min(10.0, float(duration_s))),
            "peaks": {hand: 0.0 for hand in self.hands},
        }
        return True

    def update(self, torques: dict) -> tuple[dict[str, float], dict | None]:
        now = time.time()
        ceiling = max(1e-6, float(self.config.force_haptic_max_nm))
        velocity_comp = max(0.0, float(self.config.force_haptic_velocity_comp_nm))

        for hand, state in self._states.items():
            torque_key = f"{hand}_gripper.torque"
            position_key = f"{hand}_gripper.pos"
            if torque_key not in torques:
                continue
            torque = abs(float(torques[torque_key]))
            if position_key in torques:
                position = float(torques[position_key])
                if state.previous_position is not None and state.previous_time is not None:
                    dt = max(now - state.previous_time, 1e-3)
                    velocity = (position - state.previous_position) / dt
                    state.velocity = 0.6 * state.velocity + 0.4 * velocity
                state.previous_position = position
                state.previous_time = now

            threshold = max(
                0.0,
                float(getattr(self.config, f"force_haptic_threshold_nm_{hand}")),
            ) + velocity_comp * abs(state.velocity)
            if threshold >= ceiling or not self.config.force_haptic_enabled:
                state.intensity = 0.0
            else:
                state.intensity = max(0.0, min(1.0, (torque - threshold) / (ceiling - threshold)))

            if self._calibration is not None:
                peaks = self._calibration["peaks"]
                peaks[hand] = max(peaks[hand], torque)

        result = None
        if self._calibration is not None:
            elapsed = now - self._calibration["start"]
            if elapsed >= self._calibration["duration"]:
                result = self._finish_calibration()
        return ({hand: state.intensity for hand, state in self._states.items()}, result)

    def _finish_calibration(self) -> dict:
        calibration = self._calibration
        assert calibration is not None
        peaks = calibration["peaks"]
        thresholds = {
            hand: min(1.0, peak + self.CALIBRATION_MARGIN_NM)
            for hand, peak in peaks.items()
        }
        for hand, threshold in thresholds.items():
            setattr(self.config, f"force_haptic_threshold_nm_{hand}", threshold)

        required_max = max(thresholds.values(), default=0.0) + self.MIN_CALIBRATED_SPAN_NM
        self.config.force_haptic_max_nm = max(
            float(self.config.force_haptic_max_nm), required_max
        )
        suspicious = [
            f"{hand} peak={peak:.3f} Nm"
            for hand, peak in peaks.items()
            if peak > self.SUSPICIOUS_IDLE_PEAK_NM
        ]
        self._calibration = None
        return {
            "type": "haptic_calibrate_result",
            "left_peak_nm": float(peaks.get("left", 0.0)),
            "right_peak_nm": float(peaks.get("right", 0.0)),
            "left_threshold_nm": float(thresholds.get("left", 0.0)),
            "right_threshold_nm": float(thresholds.get("right", 0.0)),
            "margin_nm": self.CALIBRATION_MARGIN_NM,
            "max_nm": float(self.config.force_haptic_max_nm),
            "suspicious": suspicious,
        }
