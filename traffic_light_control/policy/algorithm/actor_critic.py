from torch import optim
import torch
import torch.nn as nn
from policy.core import Policy
from util.type import Transition
from torch.distributions import Categorical


class ActorCritic(Policy):
    def __init__(self,
                 critic_net,
                 actor_net,
                 learning_rate,
                 discount_factor,
                 device,
                 action_space,
                 state_space) -> None:

        self.device = device
        self.critic_net = critic_net
        self.actor_net = actor_net

        self.critic_net.to(self.device)
        self.actor_net.to(self.device)

        self.critic_optim = optim.Adam(
            self.critic_net.parameters(), learning_rate)
        self.actor_optim = optim.Adam(
            self.actor_net.parameters(), learning_rate)

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.state_space = state_space
        self.critic_loss_func = nn.MSELoss()

    def compute_action(self, state, explore):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.actor_net(state)
            m = Categorical(value)
            action = m.sample()
            return action.item()

    def learn_on_batch(self, batch_data):
        batch_size = len(batch_data)
        if batch_size == 0:
            return 0.0
        actor_loss = 0.0
        critic_loss = 0.0
        for trans in batch_data:

            batch = Transition(
                torch.tensor(trans.state, dtype=torch.float).to(self.device),
                torch.tensor(trans.action, dtype=torch.long).to(self.device),
                torch.tensor(trans.reward, dtype=torch.float).to(self.device),
                torch.tensor(
                    trans.next_state, dtype=torch.float).to(self.device),
                torch.tensor(trans.done, dtype=torch.long).to(self.device)
            )
            state_batch = batch.state
            action_batch = batch.action
            reward_batch = batch.reward
            next_state_batch = batch.next_state
            mask_batch = torch.tensor(tuple(map(lambda d: not d, batch.done)),
                                      device=self.device, dtype=torch.bool)

            state_action_values = self.critic_net(state_batch)
            selected_s_a_v = state_action_values.gather(
                1, action_batch.view(-1, 1)).to(self.device)

            # 计算 critic loss
            next_action_values = torch.zeros(
                next_state_batch.shape[0], device=self.device)
            next_action_values[mask_batch] = self.critic_net(
                next_state_batch[mask_batch]).max(1)[0].detach()
            target_values = (next_action_values * self.discount_factor
                             + reward_batch)
            critic_loss += self.critic_loss_func(
                selected_s_a_v, target_values.view(-1, 1))

            # 计算 actor loss

            action_prob = self.actor_net(state_batch)
            # reward = self.__compute_reward_to_go(reward_batch)
            state_values = torch.sum(
                state_action_values.detach() * action_prob.detach(),  dim=1)
            advantage = selected_s_a_v.detach() - state_values.detach()
            m = Categorical(action_prob)
            log_prob = m.log_prob(action_batch)
            # 负数的原因是因为算法默认是梯度下降，加了负号后就可以让它变成梯度上升
            actor_loss += -torch.sum(log_prob * advantage)

        actor_loss /= batch_size
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        critic_loss /= batch_size
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss.item()

    def __compute_reward_to_go(self, rewards):
        rewards = rewards.view(-1, 1)
        weight = torch.triu(torch.ones((rewards.shape[0], rewards.shape[0]),
                                       device=self.device))
        rtg = weight.matmul(rewards)

        return rtg

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
