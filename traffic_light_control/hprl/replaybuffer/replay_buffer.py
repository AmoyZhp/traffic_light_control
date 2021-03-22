from typing import List, Union
from enum import Enum
import abc

from hprl.util.typing import MultiAgentSampleBatch, SampleBatch, SampleBatchType, Trajectory, Transition, TransitionTuple


class ReplayBufferTypes(Enum):
    Common = "Common"
    Prioritized = "PER"


class MultiAgentReplayBuffer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def store(self, data):
        ...

    @abc.abstractmethod
    def sample(self, batch_size: int) -> MultiAgentSampleBatch:
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