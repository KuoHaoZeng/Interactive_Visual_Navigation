from abc import ABC
from typing import Optional, Sequence, Union

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder
from allenact.base_abstractions.experiment_config import ExperimentConfig

from interactive_navigation.constants import THOR_COMMIT_ID

class BaseConfig(ExperimentConfig, ABC):
    PREPROCESSORS: Sequence[Union[Preprocessor, Builder[Preprocessor]]] = tuple()
    def __init__(self):
        self.SCREEN_SIZE = 224
        self.MAX_STEPS = 500
        self.ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
        self.STEP_SIZE = 0.25
        self.DISTANCE_TO_GOAL = 0.2
        self.COMMIT_ID = THOR_COMMIT_ID
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,
        }
