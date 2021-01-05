

from typing import List
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from policy.core import Policy
from util.type import Transition


class VDN(Policy):
    def __init__(self,
                 local_ids,
                 acting_nets, target_nets,
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
        self.local_ids = local_ids
        self.device = device

        self.acting_nets = acting_nets
        self.target_nets = target_nets
        params = []
        for id_, net in self.acting_nets.items():
            self.target_nets[id_].load_state_dict(
                net.state_dict())
            self.target_nets[id_].to(self.device)
            net.to(self.device)
            params.append({"params": net.parameters()})

        self.optimizer = optim.Adam(params, learning_rate)
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

    def compute_action(self, states,
                       explore: bool):

        actions = {}

        # 探索
        if explore:
            self.step = min(self.step + 1, self.eps_frame)
            self.eps = max(self.eps_init - self.step /
                           self.eps_frame, self.eps_min)
            if np.random.rand() < self.eps:
                for id_ in self.local_ids:
                    actions[id_] = np.random.choice(
                        range(self.action_space), 1).item()
                return actions
        # 利用
        for id_ in self.local_ids:
            state = torch.tensor(
                states["local"][id_], dtype=torch.float, device=self.device)
            with torch.no_grad():
                value = self.acting_nets[id_](state)
                value = np.squeeze(value, 0)
                _, index = torch.max(value, 0)
                actions[id_] = index.item()

        return actions

    def learn_on_batch(self, batch_data: List[Transition]) -> float:

        batch_size = len(batch_data)
        if batch_size == 0:
            return {
                "central": 0.0,
                "local": [],
            }
        trans_div_by_id = {}
        for id_ in self.local_ids:
            trans_div_by_id[id_] = []
        reward_batch = []
        for trans in batch_data:
            reward_batch.append(
                torch.tensor(trans.reward["central"],
                             dtype=torch.float).view(-1, 1))
            for id_ in self.local_ids:
                trans_div_by_id[id_].append(
                    Transition(
                        trans.state["local"][id_],
                        trans.action[id_],
                        0,
                        trans.next_state["local"][id_],
                        trans.done
                    )
                )
        reward_batch = torch.cat(reward_batch, 0).to(self.device)

        def np_to_torch(trans: Transition):
            torch_trans = Transition(
                torch.tensor(trans.state, dtype=torch.float),
                torch.tensor(trans.action, dtype=torch.long).view(-1, 1),
                0,
                torch.tensor(
                    trans.next_state, dtype=torch.float),
                torch.tensor(trans.done, dtype=torch.long)
            )
            return torch_trans

        central_state_action_value = torch.zeros(
            (batch_size, 1), device=self.device)
        central_next_state_action_value = torch.zeros(
            (batch_size, 1), device=self.device)

        for id_, data in trans_div_by_id.items():
            acting_net = self.acting_nets[id_]
            target_net = self.target_nets[id_]

            batch = Transition(*zip(*map(np_to_torch, data)))

            mask_batch = torch.tensor(tuple(map(lambda d: not d, batch.done)),
                                      device=self.device, dtype=torch.bool)
            state_batch = torch.cat(batch.state, 0).to(self.device)
            action_batch = torch.cat(batch.action, 0).to(self.device)
            next_state_batch = torch.cat(batch.next_state, 0).to(self.device)

            state_action_values = acting_net(
                state_batch).gather(1, action_batch).to(self.device)

            central_state_action_value += state_action_values

            next_state_action_values = torch.zeros(
                batch_size, device=self.device)
            next_state_action_values[mask_batch] = target_net(
                next_state_batch[mask_batch]).max(1)[0].detach()
            next_state_action_values = next_state_action_values.unsqueeze(1)

            central_next_state_action_value += next_state_action_values

        expected_state_action_values = (
            central_next_state_action_value *
            self.discount_factor) + reward_batch

        loss = self.loss_func(central_state_action_value,
                              expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.update_period == 0:
            for id_ in self.local_ids:
                target_net = self.target_nets[id_]
                acting_net = self.acting_nets[id_]
                target_net.load_state_dict(acting_net.state_dict())
            self.update_count = 0
        return {"central": loss.item(), "local": []}

    def get_weight(self):
        local_weight = {}
        for id_ in self.local_ids:
            local_weight[id_] = self.acting_nets[id_].state_dict()
        weight = {
            "net": local_weight,
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }
        return weight

    def set_weight(self, weight):
        optimizer_w = weight["optimizer"]
        self.optimizer.load_state_dict(optimizer_w)
        self.step = weight["step"]

        net_w = weight["net"]
        for id_ in self.local_ids:
            self.acting_nets[id_].load_state_dict(net_w[id_])
            self.target_nets[id_].load_state_dict(net_w[id_])
