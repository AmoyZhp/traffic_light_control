from typing import List, Union
import abc

from hprl.util.typing import Trajectory, Transition


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
