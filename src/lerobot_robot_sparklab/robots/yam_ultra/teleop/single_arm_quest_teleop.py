"""Single-arm adapter around `BiQuestTeleoperator`.

Strips the ``left_`` / ``right_`` prefix for one chosen arm, so a single-arm
follower expecting ``joint_1.pos`` works while the operator still wears both
controllers. Registers as ``--teleop.type=single_arm_quest_teleop``; every
BiQuest knob carries through, plus ``arm``.

Forwards `is_engaged` and `is_handoff_pressed`, OR'd across both arms. The
per-hand hooks `is_pause_pressed` / `is_reverse_pressed` are not forwarded, so
a DAgger listener runs in its combined-B/Y fallback here.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "lerobot is required to use SingleArmQuestTeleoperator."
    ) from e

from .bi_quest_teleop import BiQuestTeleoperator, BiQuestTeleoperatorConfig


@TeleoperatorConfig.register_subclass("single_arm_quest_teleop")
@dataclass
class SingleArmQuestTeleoperatorConfig(BiQuestTeleoperatorConfig):
    """Config for the single-arm Quest teleop adapter.

    Inherits every `BiQuestTeleoperatorConfig` field and adds:
    arm: which arm's action keys to forward, "left" or "right".
    """

    arm: str = "right"

    def __post_init__(self) -> None:
        if self.arm not in ("left", "right"):
            raise ValueError(
                f"SingleArmQuestTeleoperatorConfig.arm must be 'left' or 'right', "
                f"got {self.arm!r}"
            )


class SingleArmQuestTeleoperator(Teleoperator):
    """Wraps `BiQuestTeleoperator` and filters its action dict to one arm."""

    config_class = SingleArmQuestTeleoperatorConfig
    name = "single_arm_quest_teleop"

    def __init__(self, config: SingleArmQuestTeleoperatorConfig) -> None:
        super().__init__(config)
        self.config = config
        self._inner = BiQuestTeleoperator(config)
        self._prefix = f"{config.arm}_"

    # ---------- Teleoperator interface ----------

    @property
    def action_features(self) -> dict[str, type]:
        feats: dict[str, type] = {f"joint_{j}.pos": float for j in range(1, 7)}
        feats["gripper.pos"] = float
        return feats

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._inner.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._inner.is_calibrated

    def calibrate(self) -> None:
        self._inner.calibrate()

    def configure(self) -> None:
        self._inner.configure()

    def connect(self, calibrate: bool = True) -> None:
        self._inner.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self._inner.disconnect()

    def send_feedback(self, feedback: dict) -> None:
        """Prefix unprefixed torque keys with the configured arm, then forward.

        Without this BiQuest's force-haptic dispatcher cannot tell which arm a
        gripper torque belongs to.
        """
        if isinstance(feedback, dict) and isinstance(feedback.get("torques"), dict):
            relabelled: dict[str, float] = {}
            for k, v in feedback["torques"].items():
                if k.startswith("left_") or k.startswith("right_"):
                    relabelled[k] = v
                else:
                    relabelled[f"{self._prefix}{k}"] = v
            feedback = {**feedback, "torques": relabelled}
        self._inner.send_feedback(feedback)

    def publish_state(self) -> None:
        """Forward to the inner bimanual teleop. See
        :meth:`BiQuestTeleoperator.publish_state`."""
        self._inner.publish_state()

    def get_action(self) -> dict[str, float]:
        full = self._inner.get_action()
        return {k.removeprefix(self._prefix): v for k, v in full.items() if k.startswith(self._prefix)}

    # ---------- DAgger handoff hooks ----------
    # OR'd across both arms, so either hand's B/Y can signal the handoff.

    def is_engaged(self) -> bool:
        return self._inner.is_engaged()

    def is_handoff_pressed(self) -> bool:
        return self._inner.is_handoff_pressed()

    def seed_qpos_from_obs(self, obs: dict[str, float]) -> None:
        """Re-prefix a single-arm observation and forward to the inner
        bimanual teleop so its internal qpos matches the robot before a
        correction begins."""
        prefixed = {f"{self._prefix}{k}": v for k, v in obs.items()}
        self._inner.seed_qpos_from_obs(prefixed)
