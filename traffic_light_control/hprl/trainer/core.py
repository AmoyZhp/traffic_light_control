import abc
import logging
from typing import Any, Callable, Dict

from hprl.env import MultiAgentEnv
from hprl.policy import Policy
from hprl.replaybuffer import ReplayBuffer
from hprl.util.typing import TrainingRecord

Train_Fn_Type = Callable[[MultiAgentEnv, Policy,
                          ReplayBuffer, Dict, logging.Logger], Any]
Log_Record_Fn_Type = Callable[[TrainingRecord, logging.Logger], Any]


class Trainer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self, episode: int):
        ...

    @abc.abstractmethod
    def eval(self, episode: int):
        ...

    @abc.abstractmethod
    def save_checkpoint(self, checkpoint_dir: str, filename: str):
        ...

    @abc.abstractmethod
    def get_checkpoint(self):
        ...

    @abc.abstractmethod
    def set_weight(self, weight: Dict):
        ...

    @abc.abstractmethod
    def log_result(self, log_dir: str):
        ...
