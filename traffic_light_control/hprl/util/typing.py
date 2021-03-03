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
    central: float = None
    local: Dict[str, float] = field(default_factory=dict)


@dataclass
class Terminal():
    central: bool = None
    local: Dict[str, bool] = field(default_factory=dict)


@dataclass
class Transition():
    state: State
    action: Action
    reward: Reward
    next_state: State
    terminal: Terminal


@dataclass
class Trajectory():
    states: List[State]
    actions: List[Action]
    rewards: List[Reward]
    terminal: Terminal


@dataclass
class TrainingRecord():
    rewards: List[Reward] = field(default_factory=list)
    inofs: List[Dict] = field(default_factory=list)


TransitionTuple = namedtuple(
    "TransitionTuple", ["state", "action", "reward", "next_state", "terminal"])
TrajectoryTuple = namedtuple("TrajectoryTuple",
                             ["states", "actions", "rewards", "terminals"])
