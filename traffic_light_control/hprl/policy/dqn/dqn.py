from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from hprl.util.typing import SampleBatch, TransitionTuple
from hprl.policy.policy import Policy


class DQN(Policy):
    def __init__(
        self,
        acting_net: nn.Module,
        target_net: nn.Module,
        critic_lr: int,
        discount_factor: float,
        update_period: int,
        action_space,
        state_space,
        loss_fn,
        device=None,
        prioritized: bool = False,
    ) -> None:

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.acting_net = acting_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.acting_net.state_dict())

        self.target_net.to(self.device)
        self.acting_net.to(self.device)

        self.optimizer = optim.Adam(self.acting_net.parameters(), critic_lr)
        self.loss_fn = loss_fn
        self.update_count = 0
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.update_period = update_period
        self.action_space = action_space
        self.state_space = state_space
        self.prioritized = prioritized

    def compute_action(self, state: np.ndarray):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.acting_net(state)
            value = np.squeeze(value, 0)
        _, index = torch.max(value, 0)
        action = index.item()
        return action

    def learn_on_batch(self, sample_batch: SampleBatch) -> Dict:
        if sample_batch is None:
            return {}
        trans = sample_batch.transitions
        if not trans:
            return {}
        batch_size = len(trans)

        batch = np_to_torch(trans)

        mask_batch = torch.tensor(
            tuple(map(lambda d: not d, batch.terminal)),
            device=self.device,
            dtype=torch.bool,
        )

        s_batch = torch.cat(batch.state, 0).to(self.device)
        a_batch = torch.cat(batch.action, 0).to(self.device)
        r_batch = torch.cat(batch.reward, 0).to(self.device)
        next_s_batch = torch.cat(batch.next_state, 0).to(self.device)

        q_vals = self.acting_net(s_batch).gather(1, a_batch).to(self.device)

        next_q_vals = torch.zeros(batch_size, device=self.device)
        next_q_vals[mask_batch] = self.target_net(
            next_s_batch[mask_batch]).max(1)[0].detach()
        next_q_vals = next_q_vals.unsqueeze(1)

        expected_q_vals = (next_q_vals * self.discount_factor) + r_batch
        loss = 0.0
        info = {}
        if self.prioritized:
            weights = torch.tensor(sample_batch.weights, dtype=torch.float)
            weights = weights.unsqueeze(-1).to(self.device)
            loss = self.loss_fn(
                q_vals,
                expected_q_vals.detach(),
                weights.detach(),
            )
            td_error = (expected_q_vals - q_vals).squeeze(-1)
            priorities = torch.clamp(td_error, -1, 1).detach()
            priorities = abs(priorities) + 1e-6
            priorities = priorities.tolist()
            info["priorities"] = priorities
        else:
            loss = self.loss_fn(q_vals, expected_q_vals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.update_period == 0:
            self.target_net.load_state_dict(self.acting_net.state_dict())
            self.update_count = 0
        return info

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
            "critic_lr": self.critic_lr,
            "discount_factor": self.discount_factor,
            "update_period": self.update_period,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }
        return config

    def unwrapped(self):
        return self


def np_to_torch(batch_data: List[TransitionTuple]) -> TransitionTuple:
    def np_to_torch(trans: TransitionTuple):
        torch_trans = TransitionTuple(
            torch.tensor(trans.state, dtype=torch.float).unsqueeze(0),
            torch.tensor(trans.action, dtype=torch.long).view(-1, 1),
            torch.tensor(trans.reward, dtype=torch.float).view(-1, 1),
            torch.tensor(trans.next_state, dtype=torch.float).unsqueeze(0),
            torch.tensor(trans.terminal, dtype=torch.long))
        return torch_trans

    batch_data = map(np_to_torch, batch_data)
    batch = TransitionTuple(*zip(*batch_data))
    return batch
