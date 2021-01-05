from typing import List
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from util.type import Transition


class DQN():
    def __init__(self,
                 acting_net, target_net,
                 learning_rate: int,
                 discount_factor: float,
                 eps_init: float,
                 eps_min: float,
                 eps_frame: int,
                 update_period: int,
                 device,
                 action_space,
                 state_space,
                 ) -> None:
        super().__init__()
        self.device = device

        self.acting_net = acting_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.acting_net.state_dict())

        self.target_net.to(self.device)
        self.acting_net.to(self.device)

        self.optimizer = optim.Adam(
            self.acting_net.parameters(), learning_rate)
        self.loss_func = nn.MSELoss()
        self.step = 0
        self.update_count = 0

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps_frame = eps_frame
        self.update_period = update_period
        self.action_space = action_space
        self.state_space = state_space

    def compute_action(self, obs, explore: bool):
        if explore:
            self.step = min(self.step + 1, self.eps_frame)
            self.eps = max(self.eps_init - self.step /
                           self.eps_frame, self.eps_min)
            if np.random.rand() < self.eps:
                action = np.random.choice(range(self.action_space), 1).item()
                return action
        state = torch.tensor(obs, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.acting_net(state)
            value = np.squeeze(value, 0)
        _, index = torch.max(value, 0)
        action = index.item()
        return action

    def learn_on_batch(self, batch_data: List[Transition]) -> float:
        batch_size = len(batch_data)
        if batch_size == 0:
            return 0.0

        def np_to_torch(trans: Transition):
            torch_trans = Transition(
                torch.tensor(trans.state, dtype=torch.float),
                torch.tensor(trans.action, dtype=torch.long).view(-1, 1),
                torch.tensor(trans.reward, dtype=torch.float).view(-1, 1),
                torch.tensor(
                    trans.next_state, dtype=torch.float),
                torch.tensor(trans.done, dtype=torch.long)
            )
            return torch_trans

        batch_data = map(np_to_torch, batch_data)
        batch = Transition(*zip(*batch_data))

        mask_batch = torch.tensor(tuple(map(lambda d: not d, batch.done)),
                                  device=self.device, dtype=torch.bool)

        state_batch = torch.cat(batch.state, 0).to(self.device)
        action_batch = torch.cat(batch.action, 0).to(self.device)
        reward_batch = torch.cat(batch.reward, 0).to(self.device)
        next_state_batch = torch.cat(batch.next_state, 0).to(self.device)

        state_action_values = self.acting_net(
            state_batch).gather(1, action_batch).to(self.device)

        next_state_action_values = torch.zeros(batch_size, device=self.device)
        next_state_action_values[mask_batch] = self.target_net(
            next_state_batch[mask_batch]).max(1)[0].detach()
        next_state_action_values = next_state_action_values.unsqueeze(1)

        expected_state_action_values = (
            next_state_action_values * self.discount_factor) + reward_batch

        loss = self.loss_func(state_action_values,
                              expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.update_period == 0:
            self.target_net.load_state_dict(self.acting_net.state_dict())
            self.update_count = 0
        return loss.item()

    def get_weight(self):
        weight = {
            "net": self.acting_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }
        return weight

    def set_weight(self, weight):
        net_w = weight["net"]
        optimizer_w = weight["optimizer"]
        self.acting_net.load_state_dict(net_w)
        self.target_net.load_state_dict(net_w)
        self.optimizer.load_state_dict(optimizer_w)
        self.step = weight["step"]
