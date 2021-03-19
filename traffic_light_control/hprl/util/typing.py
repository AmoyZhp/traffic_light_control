from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


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
    episode: int = -1
    rewards: List[Reward] = field(default_factory=list)
    infos: List[Dict] = field(default_factory=list)

    def append_reward(self, reward: Reward):
        self.rewards.append(reward)

    def append_info(self, info: Dict):
        self.infos.append(info)

    def set_episode(self, ep: int):
        self.episode = ep


TransitionTuple = namedtuple(
    "TransitionTuple",
    ["state", "action", "reward", "next_state", "terminal"],
)
TrajectoryTuple = namedtuple(
    "TrajectoryTuple",
    ["states", "actions", "rewards", "terminal"],
)

SampleBatchType = ["SampleBatch", "MultiAgentBatch"]


@dataclass
class MultiAgentSampleBatch():
    transitions: List[Transition] = field(default_factory=list)
    trajectorys: List[Trajectory] = field(default_factory=list)
    weigths: List[Dict] = field(default_factory=list)
    idxes: List[Dict] = field(default_factory=list)


@dataclass
class SampleBatch():
    transitions: List[TransitionTuple] = field(default_factory=list)
    trajectorys: List[TrajectoryTuple] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    idxes: List[float] = field(default_factory=list)
