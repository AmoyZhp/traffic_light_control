from typing import List, Union
import abc

from hprl.util.typing import SampleBatch, SampleBatchType, Trajectory, Transition, TransitionTuple


class ReplayBuffer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def store(self, data: Union[Transition, Trajectory]):
        ...

    @abc.abstractmethod
    def sample(self,
               batch_size: int) -> Union[List[Transition], List[Trajectory]]:
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


class SingleAgentReplayBuffer(metaclass=abc.ABCMeta):
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