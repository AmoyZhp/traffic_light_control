from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List


import numpy as np

from hprl.util.enum import TrainnerTypes


@dataclass
class State():
    central: np.ndarray = None
    local: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Action():
    central: int = None
    local: Dict[str, int] = field(default_factory=dict)


@dataclass
class Reward():
    central: float = 0.0
    local: Dict[str, float] = field(default_factory=dict)


@dataclass
class Transition():
    state: State
    action: Action
    reward: Reward
    next_state: State
    terminal: bool


@dataclass
class TrainnerConfig():
    type: TrainnerTypes
    policy: Dict
    buffer: Dict
    executing: Dict


@dataclass
class DQNConfig():
    learning_rate: float
    discount_factor: float
    update_period: float
    action_space: int
    state_space: int
    eps_frame: int
    eps_init: float
    eps_min: float


@dataclass
class ReplayBufferConfig():
    capacity: int


@dataclass
class ExecConfig():
    batch_size: int


@dataclass
class TrainingRecord():
    rewards: Dict[int, Reward]


TransitionTuple = namedtuple(
    "TransitionTuple",
    ["state", "action", "reward", "next_state", "terminal"])
