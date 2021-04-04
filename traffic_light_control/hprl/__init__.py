from hprl.util.typing import State, Action, Reward, TrainingRecord, Terminal
from hprl.env import MultiAgentEnv
from hprl.env import GymWrapper
from hprl.replaybuffer import MultiAgentReplayBuffer, ReplayBufferTypes
from hprl.policy import MultiAgentPolicy, PolicyTypes, AdvantageTypes
from hprl.trainer import Trainer
from hprl.build import build_trainer, load_trainer, gym_baseline_trainer
from hprl.recorder import Recorder
import os
import logging
import hprl.recorder as recorder
import hprl.policy as policy
import hprl.env as env

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


__all__ = [
    "recorder",
    "log_to_file",
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
    "PolicyTypes",
    "ReplayBufferTypes",
    "AdvantageTypes",
    "gym_baseline_trainer",
    "Recorder",
    "policy",
    "env",
]
