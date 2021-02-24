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
class Terminal():
    central: bool = False
    local: Dict[str, bool] = field(default_factory=dict)


@dataclass
class Transition():
    state: State
    action: Action
    reward: Reward
    next_state: State
    terminal: Terminal


@dataclass
class TrainingRecord():
    rewards: Dict[int, Reward]


TransitionTuple = namedtuple(
    "TransitionTuple",
    ["state", "action", "reward", "next_state", "terminal"])
