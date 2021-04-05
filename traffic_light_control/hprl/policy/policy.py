import abc
from enum import Enum, auto
from typing import Dict

import numpy as np
from hprl.util.typing import Action, MultiAgentSampleBatch, SampleBatch, State


class PolicyTypes(Enum):
    IQL = "IQL"
    IPPO = "IPPO"
    IAC = "IAC"
    VDN = "VDN"
    COMA = "COMA"
    QMIX = "QMIX"


class AdvantageTypes(Enum):
    RewardToGO = auto()
    QMinusV = auto()


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
        batch_data: MultiAgentSampleBatch,
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
