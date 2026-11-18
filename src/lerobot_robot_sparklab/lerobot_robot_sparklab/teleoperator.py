"""The rig's teleoperation as LeRobot ``Teleoperator``s.

``--teleop.type=bi_quest_teleop``: a thin adapter over ``sparklab_yam.teleop.QuestTeleop``;
every QuestTeleopConfig knob is a ``--teleop.<name>`` flag. Needs the Quest bridge running.

``--teleop.type=bi_gello``: what the sparklab_gello nodes command, for recording; they drive the arms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray

from sparklab_yam.constants import ARM_JOINTS, HANDS
from sparklab_yam.teleop import QuestTeleop, QuestTeleopConfig

from . import ros


@TeleoperatorConfig.register_subclass("bi_quest_teleop")
@dataclass
class BiQuestTeleoperatorConfig(TeleoperatorConfig, QuestTeleopConfig):
    pass


class BiQuestTeleoperator(Teleoperator):
    config_class = BiQuestTeleoperatorConfig
    name = "bi_quest_teleop"

    def __init__(self, config: BiQuestTeleoperatorConfig) -> None:
        super().__init__(config)
        self.config = config
        self._core: QuestTeleop | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{h}_{k}.pos": float
                for h in HANDS for k in [*(f"joint_{j}" for j in range(1, ARM_JOINTS + 1)), "gripper"]}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._core is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        ros.init()
        self._node = Node("quest_teleop")
        self._core = QuestTeleop(self.config, self._node, HANDS)
        self._close_node = ros.spin_in_background(self._node)

    def disconnect(self) -> None:
        self._close_node()
        self._core = None

    def get_action(self) -> dict[str, float]:
        return self._core.step()

    def send_feedback(self, feedback: dict) -> None:
        pass

    @property
    def quest(self):
        """The raw Quest input (``sparklab_quest.input.QuestInput``)."""
        return self._core.quest

    def seed_qpos_from_obs(self, obs: dict[str, float]) -> None:
        """Re-sync the IK to `obs` and disengage. See DESIGN.md#teleoperation."""
        self._core.seed(obs)

    def xr_snapshot(self) -> dict | None:
        return self._core.xr_snapshot()


@TeleoperatorConfig.register_subclass("bi_gello")
@dataclass
class BiGelloTeleoperatorConfig(TeleoperatorConfig):
    pass


class BiGelloTeleoperator(Teleoperator):
    """The last command the GELLO nodes sent each arm, read off the drivers' command topics.

    The GELLO nodes drive the arms themselves, so the robot must not: ``--robot.command_arms=false``.
    """

    config_class = BiGelloTeleoperatorConfig
    name = "bi_gello"

    def __init__(self, config: BiGelloTeleoperatorConfig) -> None:
        super().__init__(config)
        self.config = config
        self._node: Node | None = None
        self._arm: dict[str, list[float]] = {}
        self._gripper: dict[str, float] = {}

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{h}_{k}.pos": float
                for h in HANDS for k in [*(f"joint_{j}" for j in range(1, ARM_JOINTS + 1)), "gripper"]}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._node is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True, timeout_s: float = 5.0) -> None:
        """Subscribe, and wait for a publisher on both arms' command topics, as the GELLO nodes are."""
        ros.init()
        node = self._node = Node("gello_recorder")
        for h in HANDS:
            node.create_subscription(Float64MultiArray, f"/{h}/arm/command",
                                     lambda m, h=h: self._arm.__setitem__(h, list(m.data)), 10)
            node.create_subscription(Float64, f"/{h}/gripper/command",
                                     lambda m, h=h: self._gripper.__setitem__(h, m.data), 10)
        self._close_node = ros.spin_in_background(node)
        deadline = time.monotonic() + timeout_s
        while (missing := [h for h in HANDS if not node.count_publishers(f"/{h}/arm/command")]) \
                and time.monotonic() < deadline:
            time.sleep(0.05)
        if missing:
            self.disconnect()
            raise RuntimeError(f"nothing publishes /{missing[0]}/arm/command. Start the GELLO nodes first:\n"
                               "    ros2 launch sparklab_gello teleop.launch.py cameras:=true")

    def disconnect(self) -> None:
        self._close_node()
        self._node = None

    def get_action(self) -> dict[str, float]:
        """Returns: `.pos` keys; an arm or gripper not yet commanded is left out, so the robot holds it."""
        action = {f"{h}_joint_{j + 1}.pos": v for h, q in list(self._arm.items()) for j, v in enumerate(q)}
        action.update({f"{h}_gripper.pos": g for h, g in list(self._gripper.items())})
        return action

    def send_feedback(self, feedback: dict) -> None:
        pass
