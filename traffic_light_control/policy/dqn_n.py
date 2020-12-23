from typing import List
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from util.type import Transition


class DQNNew():
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
                 ) -> None:
        super().__init__()

        self.acting_net = acting_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.acting_net.state_dict())
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
        self.device = device
        self.action_space = action_space
        self.q_value_record = []

    def compute_single_action(self, obs, explore: bool):
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
            self.q_value_record.append(value)
        _, index = torch.max(value, 0)
        action = index.item()
        return action

    def learn_on_batch(self, batch_data: List[Transition]) -> float:
        batch_size = len(batch_data)

        def np_to_torch(trans: Transition):
            torch_trans = Transition(
                torch.tensor(trans.state, dtype=torch.float),
                torch.tensor(trans.action, dtype=torch.long),
                torch.tensor(trans.reward, dtype=torch.float),
                torch.tensor(
                    trans.next_state, dtype=torch.float),
                torch.tensor(trans.done, dtype=torch.long)
            )
            return torch_trans

        batch_data = map(np_to_torch, batch_data)
        batch = Transition(*zip(*batch_data))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        valid_next_state_batch = [s for s in batch.next_state if s is not None]
        non_final_next_states = None
        if len(valid_next_state_batch) > 0:
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]).to(self.device)

        state_batch = torch.cat(batch.state, 0).to(self.device)
        action_batch = torch.cat(batch.action, 0).to(self.device)
        reward_batch = torch.cat(batch.reward, 0).to(self.device)

        state_action_values = self.acting_net(
            state_batch).gather(1, action_batch).to(self.device)

        next_state_values = torch.zeros(batch_size, device=self.device)
        if non_final_next_states is not None:
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1)[0].detach()
        next_state_values = next_state_values.unsqueeze(1)

        expected_state_action_values = (
            next_state_values * self.discount_factor) + reward_batch

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
        }
        return weight

    def set_weight(self, weight):
        net_w = weight["net"]
        optimizer_w = weight["optimizer"]
        self.acting_net.load_state_dict(net_w)
        self.target_net.load_state_dict[net_w]
        self.optimizer.load_state_dict(optimizer_w)

    def get_config(self):
        config = {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "eps_init": self.eps_init,
            "eps_min": self.eps_min,
            "eps_frame": self.eps_frame,
            "update_period": self.update_period,
            "step": self.step
        }
        return config
