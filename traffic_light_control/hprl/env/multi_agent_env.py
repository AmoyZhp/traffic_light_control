import abc
from typing import Dict, List, Tuple

from hprl.util.typing import Action, Reward, State, Terminal


class MultiAgentEnv(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, actions: Action) -> Tuple[State, Reward, Terminal, Dict]:
        ...

    @abc.abstractmethod
    def reset(self) -> State:
        ...

    @property
    @abc.abstractmethod
    def agents_id(self) -> List[str]:
        ...

    @property
    @abc.abstractmethod
    def central_action_space(self):
        ...

    @property
    @abc.abstractmethod
    def local_action_space(self):
        ...

    @property
    @abc.abstractmethod
    def central_state_space(self):
        ...

    @property
    @abc.abstractmethod
    def local_state_space(self):
        ...

    @property
    @abc.abstractmethod
    def setting(self):
        ...

    @property
    @abc.abstractmethod
    def name(self):
        ...