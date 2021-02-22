import abc
from typing import Any, Callable

from hprl.env import MultiAgentEnv
from hprl.policy import Policy
from hprl.replaybuffer import ReplayBuffer

Train_Fn_Type = Callable[[MultiAgentEnv, Policy, ReplayBuffer], Any]


class Trainer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self, episode: int):
        ...

    @abc.abstractmethod
    def eval(self, episode: int):
        ...

    @abc.abstractmethod
    def save_checkpoint(self, checkpoint_dir: str):
        ...

    @abc.abstractmethod
    def load_checkpoint(self, checkpoint_file: str):
        ...

    @abc.abstractmethod
    def log_result(self, log_dir: str):
        ...
