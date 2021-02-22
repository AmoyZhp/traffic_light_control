import abc
from typing import Dict, List, Tuple

from hprl.util.typing import Action, Reward, State


class MultiAgentEnv(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def step(self,
             actions: Action
             ):
        ...

    @abc.abstractmethod
    def reset(self) -> State:
        ...

    @abc.abstractmethod
    def get_agents_id(self) -> List[str]:
        ...
