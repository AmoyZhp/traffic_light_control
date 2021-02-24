from typing import List
import abc

from hprl.util.typing import Transition


class ReplayBuffer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def store(self, transition: Transition):
        ...

    @abc.abstractmethod
    def sample(self, batch_size: int) -> List[Transition]:
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
