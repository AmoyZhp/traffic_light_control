from typing import List
from torch import optim
import torch
import torch.nn as nn
from policy.core import Policy
from util.type import Transition
from torch.distributions import Categorical


class PPO(Policy):
    def __init__(self,
                 critic_net: nn.Module,
                 critic_target_net: nn.Module,
                 actor_net: nn.Module,
                 inner_epoch: int,
                 learning_rate: float,
                 discount_factor: float,
                 update_period: int,
                 device: torch.device,
                 action_space,
                 state_space,
                 clip_param=0.2,) -> None:

        self.device = device

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

    def compute_action(self, state, explore):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.actor_net(state)
            m = Categorical(value)
            action = m.sample()
            return action.item()

    def __inner_train(self, state_seq_batch, action_seq_batch,
                      reward_seq_batch, sel_old_action_prob):
        # seq * batch * action_space
        q_vals = self.critic_net(state_seq_batch)

        # seq * batch * 1
        selected_q_v = q_vals.gather(
            2, action_seq_batch).to(self.device)

        # 计算 critic loss

        # seq * batch * 1
        next_sel_q_v = torch.zeros_like(
            selected_q_v, device=self.device)
        next_sel_q_v[:-1] = self.critic_target_net(
            state_seq_batch[1:]).gather(
            2, action_seq_batch[1:])
        target_values = (next_sel_q_v * self.discount_factor
                         + reward_seq_batch)
        critic_loss = self.__compute_critic_loss(
            selected_q_v, target_values.detach())

        # 计算 actor loss

        # seq * batch * action_shape
        action_prob = self.actor_net(state_seq_batch)
        sel_action_prob = action_prob.gather(2, action_seq_batch)

        state_values = torch.sum(
            q_vals * action_prob,
            dim=2).unsqueeze(-1)
        advantage = selected_q_v - state_values
        advantage = advantage.detach()

        ratio = sel_action_prob / sel_old_action_prob
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_param,
                            1 + self.clip_param) * advantage

        # 负数的原因是因为算法默认是梯度下降，加了负号后就可以让它变成梯度上升
        actor_loss = -torch.min(surr1, surr2).mean()

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
        return critic_loss.item()

    def learn_on_batch(self, batch_data: List[Transition]):
        batch_size = len(batch_data)
        if batch_size == 0:
            return 0.0

        def np_to_tensor(trans: Transition):
            return Transition(
                torch.tensor(trans.state, dtype=torch.float).to(
                    self.device).unsqueeze(1),
                torch.tensor(trans.action, dtype=torch.long).to(
                    self.device).view(-1, 1).unsqueeze(1),
                torch.tensor(trans.reward, dtype=torch.float).to(
                    self.device).view(-1, 1).unsqueeze(1),
                torch.tensor(
                    trans.next_state, dtype=torch.float).to(self.device),
                torch.tensor(trans.done, dtype=torch.long).to(self.device)
            )
        batch_data = map(np_to_tensor, batch_data)
        batch_data = Transition(*zip(*batch_data))

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

    def __compute_reward_to_go(self, rewards):
        rewards = rewards.view(-1, 1)
        weight = torch.triu(torch.ones((rewards.shape[0], rewards.shape[0]),
                                       device=self.device))
        rtg = weight.matmul(rewards)

        return rtg

    def __compute_critic_loss(self, input, target):
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
