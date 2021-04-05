import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hprl.policy.policy import Policy
from hprl.util.typing import SampleBatch, TransitionTuple

logger = logging.getLogger(__name__)


class DQN(Policy):
    def __init__(
        self,
        acting_net: nn.Module,
        target_net: nn.Module,
        learning_rate: int,
        discount_factor: float,
        update_period: int,
        action_space,
        state_space,
        loss_fn,
        device=None,
        prioritized: bool = False,
    ) -> None:

        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
        self._acting_net = acting_net
        self._target_net = target_net
        self._target_net.load_state_dict(self._acting_net.state_dict())

        self._target_net.to(self._device)
        self._acting_net.to(self._device)

        self._optimizer = optim.Adam(self._acting_net.parameters(),
                                     learning_rate)
        self._loss_fn = loss_fn
        self._update_count = 0
        self._lr = learning_rate
        self._gamma = discount_factor
        self._update_period = update_period
        self._action_space = action_space
        self._state_space = state_space
        self._prioritized = prioritized

        logger.info("DQN init")
        logger.info("\t learning rate : %f", self._lr)
        logger.info("\t discount factor : %f", self._gamma)
        logger.info("\t update period : %d", self._update_period)
        logger.info("\t state space : %d", self._state_space)
        logger.info("\t action space : %d", self._action_space)
        logger.info("\t prioritized : %s", self._prioritized)
        logger.info("DQN init end")

    def compute_action(self, state: np.ndarray):
        state = torch.tensor(state, dtype=torch.float, device=self._device)
        with torch.no_grad():
            value = self._acting_net(state)
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
            device=self._device,
            dtype=torch.bool,
        )

        s_batch = torch.cat(batch.state, 0).to(self._device)
        a_batch = torch.cat(batch.action, 0).to(self._device)
        r_batch = torch.cat(batch.reward, 0).to(self._device)
        next_s_batch = torch.cat(batch.next_state, 0).to(self._device)

        q_vals = self._acting_net(s_batch).gather(1, a_batch).to(self._device)

        next_q_vals = torch.zeros(batch_size, device=self._device)
        next_q_vals[mask_batch] = self._target_net(
            next_s_batch[mask_batch]).max(1)[0].detach()
        next_q_vals = next_q_vals.unsqueeze(1)

        expected_q_vals = (next_q_vals * self._gamma) + r_batch
        loss = 0.0
        info = {}
        if self._prioritized:
            weights = torch.tensor(sample_batch.weights, dtype=torch.float)
            weights = weights.unsqueeze(-1).to(self._device)
            loss = self._loss_fn(
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
            loss = self._loss_fn(q_vals, expected_q_vals)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._update_count += 1
        if self._update_count % self._update_period == 0:
            self._target_net.load_state_dict(self._acting_net.state_dict())
            self._update_count = 0
        return info

    def get_weight(self):
        weight = {
            "net": self._acting_net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        return weight

    def set_weight(self, weight):
        net_w = weight["net"]
        optimizer_w = weight["optimizer"]
        self._acting_net.load_state_dict(net_w)
        self._target_net.load_state_dict(net_w)
        self._optimizer.load_state_dict(optimizer_w)
        self.step = weight["step"]

    def get_config(self):
        config = {
            "critic_lr": self._lr,
            "discount_factor": self._gamma,
            "update_period": self._update_period,
            "action_space": self._action_space,
            "state_space": self._state_space,
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
