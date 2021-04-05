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


def log_to_file(path: str = ""):
    if not path:
        path = "hprl.log"
    elif os.path.isdir(path):
        path = f"{path}/hprl.log"
    filehander = logging.FileHandler(path, "a")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", )
    filehander.setFormatter(formatter)
    logger.addHandler(filehander)


from hprl.build import build_trainer, gym_baseline_trainer, load_trainer
from hprl.env import GymWrapper, MultiAgentEnv
from hprl.policy import AdvantageTypes, MultiAgentPolicy, PolicyTypes
from hprl.replaybuffer import (MAgentReplayBuffer, ReplayBuffer,
                               ReplayBufferTypes)
from hprl.trainer import Trainer
from hprl.util.typing import Action, Reward, State, Terminal, TrainingRecord

__all__ = [
    "build_trainer",
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
    "gym_baseline_trainer",
]
