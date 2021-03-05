import abc
from typing import Dict, List, Union
from hprl.util.typing import Action, State, Trajectory, Transition


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_action(self, state: State) -> Action:
        ...

    @abc.abstractmethod
    def learn_on_batch(
        self,
        batch_data: Union[List[Transition], List[Trajectory]],
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
    def unwrapped(self) -> "Policy":
        ...
