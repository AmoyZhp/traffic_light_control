from hprl.util.typing import Action, State, Trajectory, Transition
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from hprl.policy.core import Policy


class PPO(Policy):
    def __init__(self,
                 critic_net: nn.Module,
                 critic_target_net: nn.Module,
                 actor_net: nn.Module,
                 inner_epoch: int,
                 learning_rate: float,
                 discount_factor: float,
                 update_period: int,
                 action_space,
                 state_space,
                 clip_param=0.2,) -> None:

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.actor_net = actor_net
        self.critic_net = critic_net
        self.critic_target_net = critic_target_net

        self.critic_target_net.load_state_dict(self.critic_net.state_dict())

        self.critic_net.to(self.device)
        self.critic_target_net.to(self.device)
        self.actor_net.to(self.device)

        self.critic_optim = optim.Adam(
            self.critic_net.parameters(), learning_rate)
        self.actor_optim = optim.Adam(
            self.actor_net.parameters(), learning_rate)

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.state_space = state_space
        self.update_count = 0
        self.update_period = update_period
        self.inner_epoch = inner_epoch
        self.clip_param = clip_param

    def compute_action(self, state: State) -> Action:
        state = torch.tensor(
            state.central, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.actor_net(state)
            m = Categorical(value)
            action = m.sample()
            return Action(central=action.item())

    def learn_on_batch(self, batch_data: List[Trajectory]):
        batch_size = len(batch_data)
        if batch_size == 0:
            return 0.0

        batch = self._to_torch_batch(batch_data)

        state_seq_batch = torch.cat(batch_data.state, 1)
        action_seq_batch = torch.cat(batch_data.action, 1)
        reward_seq_batch = torch.cat(batch_data.reward, 1)

        sel_old_action_prob = self.actor_net(
            state_seq_batch).gather(2, action_seq_batch).detach()
        total_loss = 0
        for _ in range(self.inner_epoch):
            total_loss += self.__inner_train(
                state_seq_batch=state_seq_batch,
                action_seq_batch=action_seq_batch,
                reward_seq_batch=reward_seq_batch,
                sel_old_action_prob=sel_old_action_prob
            )
        return total_loss / self.inner_epoch

    def _inner_loop(self):
        ...

    def _critic_loss(self, input, target):
        return torch.mean((input - target) ** 2)

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
        return super().get_config()

    def unwrapped(self):
        return self
