import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s", )

stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

from hprl.build import load_trainer, log_to_file
from hprl.env import GymWrapper, MultiAgentEnv
from hprl.policy import MultiAgentPolicy
from hprl.replaybuffer import MAgentReplayBuffer, ReplayBuffer
from hprl.trainer import Trainer
from hprl.typing import (Action, AdvantageTypes, PolicyTypes,
                         ReplayBufferTypes, Reward, State, Terminal,
                         TrainingRecord)

__all__ = [
    "load_trainer",
    "log_to_file",
    "Trainer",
    "Policy",
    "MultiAgentPolicy",
    "MAgentReplayBuffer",
    "ReplayBuffer",
    "MultiAgentEnv",
    "State",
    "Action",
    "Reward",
    "Terminal",
    "TrainingRecord",
    "GymWrapper",
    "PolicyTypes",
    "ReplayBufferTypes",
    "AdvantageTypes",
]
