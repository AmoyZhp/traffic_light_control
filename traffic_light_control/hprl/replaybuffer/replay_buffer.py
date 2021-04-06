import abc
from enum import Enum
from typing import Dict, List, Union

from hprl.typing import (MultiAgentSampleBatch, SampleBatch, Transition,
                         TransitionTuple)


class MAgentReplayBuffer(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def type(self):
        ...

    @abc.abstractmethod
    def store(
        self,
        data: Transition,
        priorities: Union[float, Dict[str, float]],
    ):
        ...

    @abc.abstractmethod
    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> Union[MultiAgentSampleBatch, Dict[str, SampleBatch]]:
        ...

    @abc.abstractmethod
    def update_priorities(
        self,
        idxes: Union[List[float], Dict[str, List[float]]],
        priorities: Union[List[float], Dict[str, List[float]]],
    ):
        ...

    @abc.abstractmethod
    def clear(self):
        ...

    @abc.abstractmethod
    def get_weight(self) -> Dict:
        ...

    @abc.abstractmethod
    def set_weight(self, weight: Dict):
        ...

    @abc.abstractmethod
    def get_config(self):
        ...


class ReplayBuffer(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def type(self):
        ...

    @abc.abstractmethod
    def store(self, data: TransitionTuple, priorities: float):
        ...

    @abc.abstractmethod
    def sample(self, batch_size: int, beta: float) -> SampleBatch:
        ...

    @abc.abstractmethod
    def update_priorities(self, idxes: List[int], priorities: List[float]):
        ...

    @abc.abstractmethod
    def clear(self):
        ...

    @abc.abstractmethod
    def get_weight(self):
        ...

    @abc.abstractmethod
    def set_weight(self, weight):
        ...

    @abc.abstractmethod
    def get_config(self):
        ...
