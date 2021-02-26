from typing import List


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from hprl.util.typing import Action, State, Transition, TransitionTuple, TrainnerTypes
from hprl.policy.core import Policy
from hprl.util.enum import ReplayBufferTypes
from hprl.policy.nets import CartPole


def get_default_config():
    capacity = 4000
    learning_rate = 1e-3
    batch_size = 16
    discount_factor = 0.99
    eps_init = 1.0
    eps_min = 0.01
    eps_frame = 2000
    update_period = 50
    action_space = 2
    state_space = 4

    policy_config = {
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "update_period": update_period,
        "action_space": action_space,
        "state_space": state_space,
        "eps_frame": eps_frame,
        "eps_init": eps_init,
        "eps_min": eps_min,
    }
    buffer_config = {
        "type": ReplayBufferTypes.Common,
        "capacity": capacity,
    }
    exec_config = {
        "batch_size": batch_size,
        "base_dir": "records",
        "check_frequency": 100,
    }
    trainner_config = {
        "type": TrainnerTypes.IQL,
        "executing": exec_config,
        "policy": policy_config,
        "buffer": buffer_config,
    }

    acting_net = CartPole(
        input_space=state_space,
        output_space=action_space
    )

    target_net = CartPole(
        input_space=state_space,
        output_space=action_space
    )

    model = {
        "acting": acting_net,
        "target": target_net,
    }

    return trainner_config, model


class DQN(Policy):
    def __init__(self,
                 acting_net: nn.Module,
                 target_net: nn.Module,
                 learning_rate: int,
                 discount_factor: float,
                 update_period: int,
                 action_space,
                 state_space,
                 ) -> None:
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.acting_net = acting_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.acting_net.state_dict())

        self.target_net.to(self.device)
        self.acting_net.to(self.device)

        self.optimizer = optim.Adam(
            self.acting_net.parameters(), learning_rate)
        self.loss_func = nn.MSELoss()
        self.update_count = 0
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.update_period = update_period
        self.action_space = action_space
        self.state_space = state_space

    def compute_action(self, state: State):
        state = torch.tensor(
            state.central, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.acting_net(state)
            value = np.squeeze(value, 0)
        _, index = torch.max(value, 0)
        action = index.item()
        return Action(central=action)

    def learn_on_batch(self, batch_data: List[Transition]) -> float:
        if batch_data == None or len(batch_data) == 0:
            return
        batch_size = len(batch_data)

        batch = self._to_torch_batch(batch_data)

        mask_batch = torch.tensor(tuple(map(lambda d: not d, batch.terminal)),
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

    def _to_torch_batch(self, batch_data: List[Transition]):

        def np_to_torch(trans: Transition):
            torch_trans = TransitionTuple(
                torch.tensor(trans.state.central,
                             dtype=torch.float).unsqueeze(0),
                torch.tensor(trans.action.central,
                             dtype=torch.long).view(-1, 1),
                torch.tensor(trans.reward.central,
                             dtype=torch.float).view(-1, 1),
                torch.tensor(
                    trans.next_state.central, dtype=torch.float).unsqueeze(0),
                torch.tensor(trans.terminal.central, dtype=torch.long)
            )
            return torch_trans

        batch_data = map(np_to_torch, batch_data)
        batch = TransitionTuple(*zip(*batch_data))
        return batch

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
        self.target_net.load_state_dict(net_w)
        self.optimizer.load_state_dict(optimizer_w)
        self.step = weight["step"]

    def get_config(self):
        config = {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "update_period": self.update_period,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }
        return config

    def unwrapped(self):
        return self
