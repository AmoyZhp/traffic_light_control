import abc
from typing import Dict, List, Union

import torch

from hprl.util.typing import Action, State, Trajectory, Transition


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_action(self, state: State) -> Action:
        ...

    @abc.abstractmethod
    def learn_on_batch(self, batch_data: Union[List[Transition],
                                               List[Trajectory]]):
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

    @abc.abstractmethod
    def unwrapped(self):
        ...


def to_tensor_for_trajectory(batch_data: List[Trajectory], device=None):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def np_to_tensor(data: Trajectory):
        states = map(
            lambda s: torch.tensor(
                s.central, dtype=torch.float).unsqueeze(0).to(device),
            data.states,
        )
        actions = map(
            lambda a: torch.tensor(
                a.central, dtype=torch.long).view(-1, 1).to(device),
            data.actions
        )
        rewards = map(
            lambda r: torch.tensor(
                r.central, dtype=torch.float).view(-1, 1).to(device),
            data.rewards
        )
        return Trajectory(
            states=list(states),
            actions=list(actions),
            rewards=list(rewards),
            terminal=data.terminal
        )
    batch_data = list(map(np_to_tensor, batch_data))
    states = []
    actions = []
    rewards = []
    for data in batch_data:
        # cated shape is 1 * seq_len * data_space
        seq_s = torch.cat(data.states, 0).unsqueeze(0)
        seq_a = torch.cat(data.actions, 0).unsqueeze(0)
        seq_r = torch.cat(data.rewards, 0).unsqueeze(0)
        states.append(seq_s)
        actions.append(seq_a)
        rewards.append(seq_r)

    # why not cat here because they may have not equal length
    return states, actions, rewards


def compute_reward_to_go(rewards: torch.tensor, device=None):
        # reward shape is : batch * seq_len * 1
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.triu(torch.ones(
        (rewards.shape[0],
         rewards.shape[1],
         rewards.shape[1]))).to(device)
    rtg = weight.matmul(rewards)

    return rtg
