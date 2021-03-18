import abc
import numpy as np
from typing import Dict, List, Union
from hprl.util.typing import Action, MultiAgentBatch, SampleBatch, State, Trajectory, Transition


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_action(self, state: np.ndarray) -> int:
        ...

    @abc.abstractmethod
    def learn_on_batch(self, batch_data: SampleBatch) -> Dict:
        ...

    @abc.abstractmethod
    def get_weight(self) -> Dict:
        ...

    @abc.abstractmethod
    def set_weight(self, weight: Dict):
        ...

    @abc.abstractmethod
    def get_config(self) -> Dict:
        ...

    @abc.abstractmethod
    def unwrapped(self) -> "Policy":
        ...


class MultiAgentPolicy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_action(self, state: State) -> Action:
        ...

    @abc.abstractmethod
    def learn_on_batch(
        self,
        batch_data: MultiAgentBatch,
    ):
        ...

    @abc.abstractmethod
    def get_weight(self) -> Dict:
        ...

    @abc.abstractmethod
    def set_weight(self, weight: Dict):
        ...

    @abc.abstractmethod
    def get_config(self) -> Dict:
        ...

    @abc.abstractmethod
    def unwrapped(self) -> "MultiAgentPolicy":
        ...
