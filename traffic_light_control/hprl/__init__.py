from hprl.util.typing import State, Action, Reward, TrainingRecord, Terminal
from hprl.env import MultiAgentEnv
from hprl.env import GymWrapper
from hprl.replaybuffer import MultiAgentReplayBuffer
from hprl.policy import MultiAgentPolicy
from hprl.trainer import Trainer
from hprl.util.enum import TrainnerTypes, ReplayBufferTypes, AdvantageTypes
from hprl.build import build_trainer, load_trainer, test_trainer

import logging

__all__ = [
    "load_trainer",
    "build_trainer",
    "Trainer",
    "Policy",
    "MultiAgentPolicy",
    "MultiAgentReplayBuffer",
    "ReplayBuffer",
    "MultiAgentEnv",
    "State",
    "Action",
    "Reward",
    "Terminal",
    "TrainingRecord",
    "GymWrapper",
    "TrainnerTypes",
    "ReplayBufferTypes",
    "AdvantageTypes",
    "test_trainer",
]
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", )

stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)