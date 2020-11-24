import numpy as np
import torch
from torch.autograd import backward
import torch.nn as nn
import torch.optim as optim
from policy.buffer.replay_memory import ReplayMemory, Transition
import policy.net.single_intersection as net


class DQNConfig():
    def __init__(self, learning_rate: float, batch_size: float, capacity: int,
                 discount_factor: float, eps_init: float, eps_min: float,
                 eps_frame: int, update_count: int,
                 action_space, state_space, device) -> None:
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps_frame = eps_frame
        self.update_count = update_count
        self.device = device
        self.action_space = action_space
        self.state_space = state_space


class DQN():
    def __init__(self, memory: ReplayMemory,
                 target_net: net.SingleIntesection,
                 acting_net: net.SingleIntesection, config: DQNConfig) -> None:
        super(DQN, self).__init__()
        self.device = config.device
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.discount_factor = config.discount_factor
        self.eps_init = config.eps_init
        self.eps_min = config.eps_min
        self.eps_frame = config.eps_frame
        self.eps = self.eps_init
        self.update_count = config.update_count
        self.action_space = config.action_space
        self.state_space = config.state_space

        self.memory = memory
        self.acting_net = acting_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.acting_net.state_dict())

        self.target_net.to(self.device)
        self.acting_net.to(self.device)
        self.optimizer = optim.Adam(
            self.acting_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.step = 0

    def select_action(self, state):
        self.step += 1
        self.eps = max(self.eps_init - self.step /
                       self.eps_frame, self.eps_min)
        if np.random.rand() < self.eps:
            action = np.random.choice(range(self.action_space), 1).item()
            return action
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.acting_net(state)
        _, index = torch.max(value, 0)
        action = index.item()
        return action

    def select_eval_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.acting_net(state)
        _, index = torch.max(value, 0)
        action = index.item()
        return action

    def store_transition(self, transition):
        self.memory.sotre(transition)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        trans_batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*trans_batch))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        valid_next_state_batch = [s for s in batch.next_state if s is not None]
        non_final_next_states = None
        if len(valid_next_state_batch) > 0:
            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None
                                               ]).to(self.device)
        else:
            print(batch.next_state)

        state_batch = torch.cat(batch.state, 0).to(self.device)
        action_batch = torch.cat(batch.action, 0).to(self.device)
        reward_batch = torch.cat(batch.reward, 0).to(self.device)

        state_action_values = self.acting_net(
            state_batch).gather(1, action_batch).to(self.device)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
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
        if self.update_count % self.update_count == 0:
            self.target_net.load_state_dict(self.acting_net.state_dict())

    def save_model(self, path: str):
        state = {"net": self.acting_net.state_dict(),
                 "optimizer": self.optimizer.state_dict(), }
        torch.save(state, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.acting_net.load_state_dict(checkpoint["net"])
        self.target_net.load_state_dict(checkpoint["net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
