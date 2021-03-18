import logging
from typing import List
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch import optim

from hprl.policy.policy import Policy
from hprl.policy.util import to_tensor_for_trajectory, compute_reward_to_go
from hprl.util.enum import AdvantageTypes
from hprl.util.typing import SampleBatch

logger = logging.getLogger(__name__)


class ActorCritic(Policy):
    def __init__(
            self,
            critic_net: nn.Module,
            critic_target_net: nn.Module,
            actor_net: nn.Module,
            critic_lr: float,
            actor_lr: float,
            discount_factor: float,
            update_period: int,
            action_space,
            state_space,
            critic_loss_fn,
            device: torch.device = None,
            advantage_type: AdvantageTypes = AdvantageTypes.RewardToGO
    ) -> None:
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.actor_net = actor_net
        self.critic_net = critic_net
        self.critic_target_net = critic_target_net

        self.critic_target_net.load_state_dict(self.critic_net.state_dict())

        self.critic_net.to(self.device)
        self.critic_target_net.to(self.device)
        self.actor_net.to(self.device)

        self.critic_optim = optim.Adam(self.critic_net.parameters(), critic_lr)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), actor_lr)

        self.discount_factor = discount_factor
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.action_space = action_space
        self.state_space = state_space
        self.update_count = 0
        self.update_period = update_period
        self.critic_loss_fn = critic_loss_fn
        self.advantage_type = advantage_type

    def compute_action(self, state: np.ndarray) -> int:
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.actor_net(state)
            m = Categorical(value)
            action = m.sample()
            return action.item()

    def learn_on_batch(self, sample_batch: SampleBatch):
        if sample_batch is None:
            return
        trajectories = sample_batch.trajectorys
        if not trajectories:
            return
        batch_size = len(trajectories)
        traj_len = len(trajectories[0].states)
        traj_len_equal = True
        for traj in trajectories:
            if traj_len != len(traj.states):
                traj_len_equal = False
                break

        states, actions, rewards = to_tensor_for_trajectory(trajectories)
        critic_loss = 0.0
        actor_loss = 0.0
        if traj_len_equal:
            # if they have equal sequence length
            # they could be cated in batch dim
            # data shape : batch_size * seq_len * data_shape
            cat_states = torch.cat(states, 0).to(self.device)
            cat_actions = torch.cat(actions, 0).to(self.device)
            cat_rewards = torch.cat(rewards, 0).to(self.device)
            critic_loss, actor_loss = self._traj_len_equal_learn(
                states=cat_states,
                actions=cat_actions,
                rewards=cat_rewards,
            )
        else:
            for i in range(batch_size):
                # because we learn element independetly
                # each element are 1 * seq_len * data_shape
                # so could use equal learn method
                critic_loss_temp, actor_loss_temp = self._traj_len_equal_learn(
                    states=states[i],
                    actions=actions[i],
                    rewards=rewards[i],
                )
                critic_loss += critic_loss_temp
                actor_loss += actor_loss_temp

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.update_count += 1
        if self.update_count > self.update_period:
            self.update_count = 0
            self.critic_target_net.load_state_dict(
                self.critic_net.state_dict())
        return {}

    def _traj_len_equal_learn(self, states: torch.tensor,
                              actions: torch.tensor, rewards: torch.tensor):
        # states; actions and rewards shape
        # are batch_size * seq_length * data_space
        states.to(self.device)
        actions.to(self.device)
        rewards.to(self.device)

        q_vals = self.critic_net(states)

        # shape : batch * seq * 1
        selected_q_v = q_vals.gather(2, actions).to(self.device)

        # shape : batch * seq * 1
        next_sel_q_v = torch.zeros_like(selected_q_v, device=self.device)
        next_sel_q_v[:-1] = self.critic_target_net(states[1:]).gather(
            2, actions[1:])
        target_values = (next_sel_q_v * self.discount_factor + rewards)
        critic_loss = self.critic_loss_fn(selected_q_v, target_values.detach())

        advantage = 0.0
        action_prob = self.actor_net(states)
        if self.advantage_type == AdvantageTypes.QMinusV:
            state_values = torch.sum(q_vals * action_prob, dim=2).unsqueeze(-1)
            advantage = selected_q_v - state_values
        elif self.advantage_type == AdvantageTypes.RewardToGO:
            advantage = compute_reward_to_go(rewards,
                                             self.device) - selected_q_v
        else:
            raise ValueError(f"advantage type invalid {self.advantage_type}")
        log_prob = torch.log(action_prob).gather(2, actions)
        # because default alg is gradient desecent.
        # negative turn it into gradient ascendent
        actor_loss = (-torch.mean(log_prob * advantage.detach()))
        return critic_loss, actor_loss

    def get_weight(self):
        weight = {
            "net": {
                "actor": self.actor_net.state_dict(),
                "critic": self.critic_net.state_dict(),
            },
            "optimizer": {
                "actor": self.actor_optim.state_dict(),
                "critic": self.critic_optim.state_dict(),
            }
        }
        return weight

    def set_weight(self, weight):
        net_w = weight["net"]
        optimizer_w = weight["optimizer"]
        self.actor_net.load_state_dict(net_w["actor"])
        self.critic_net.load_state_dict(net_w["critic"])
        self.actor_optim.load_state_dict(optimizer_w["actor"])
        self.critic_optim.load_state_dict(optimizer_w["critic"])

    def get_config(self):
        config = {
            "critic_lr": self.critic_lr,
            "actor_lr": self.actor_lr,
            "discount_factor": self.discount_factor,
            "update_period": self.update_period,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }
        return config

    def unwrapped(self):
        return self
