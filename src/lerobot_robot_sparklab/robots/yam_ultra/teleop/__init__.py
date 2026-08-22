"""LeRobot ``Teleoperator`` adapters.

Importing these registers the config types, after which
``--teleop.type=bi_quest_teleop`` / ``single_arm_quest_teleop`` work.
"""

from .bi_quest_teleop import BiQuestTeleoperator, BiQuestTeleoperatorConfig
from .single_arm_quest_teleop import (
    SingleArmQuestTeleoperator,
    SingleArmQuestTeleoperatorConfig,
)

__all__ = [
    "BiQuestTeleoperator",
    "BiQuestTeleoperatorConfig",
    "SingleArmQuestTeleoperator",
    "SingleArmQuestTeleoperatorConfig",
]
