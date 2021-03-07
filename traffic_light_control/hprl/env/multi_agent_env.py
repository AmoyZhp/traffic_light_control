import abc
from typing import Dict, List, Tuple

from hprl.util.typing import Action, Reward, State, Terminal


class MultiAgentEnv(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def step(self,
             actions: Action
             ) -> Tuple[State, Reward, Terminal, Dict]:
        ...

    @abc.abstractmethod
    def reset(self) -> State:
        ...

    @abc.abstractmethod
    def get_agents_id(self) -> List[str]:
        ...

    @abc.abstractmethod
    def get_central_action_space(self):
        ...

    @abc.abstractmethod
    def get_local_action_space(self):
        ...

    @abc.abstractmethod
    def get_central_state_space(self):
        ...

    @abc.abstractmethod
    def get_local_state_space(self):
        ...
