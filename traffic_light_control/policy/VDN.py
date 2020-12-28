

from typing import List
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from util.type import Transition


class VDN():
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
        for id_, net in self.acting_nets:
            self.target_nets[id_].load_state_dict(
                net.state_dict())
            self.target_nets[id_].to(self.device)
            net.to(self.device)
            params.append({"params": net.parameters})

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
                states[id_], dtype=torch.float, device=self.device)
            with torch.no_grad():
                actions[id_] = self.acting_nets[id_](state)

        return actions

    def learn_on_batch(self, batch_data: List[Transition]) -> float:

        batch_size = len(batch_data)

        trans_div_by_id = {}
        for id_ in self.local_ids:
            trans_div_by_id[id_] = []
        reward_batch = []
        for trans in batch_data:
            reward_batch.append(
                torch.tensor(trans.reward, dtype=torch.float))
            for id_ in self.local_ids:
                trans_div_by_id[id_].append(
                    Transition(
                        trans.state[id_],
                        trans.action[id_],
                        0,
                        trans.next_state[id_],
                        trans.done
                    )
                )
        reward_batch = torch.cat(reward_batch, 0).to(self.device)

        def np_to_torch(trans: Transition):
            torch_trans = Transition(
                torch.tensor(trans.state, dtype=torch.float),
                torch.tensor(trans.action, dtype=torch.long),
                0,
                torch.tensor(
                    trans.next_state, dtype=torch.float),
                torch.tensor(trans.done, dtype=torch.long)
            )
            return torch_trans

        true_state_values = torch.zeros((batch_size, 1))
        true_next_state_values = torch.zeros((batch_size, 1))
        for id_, data in trans_div_by_id.item():
            batch = Transition(*zip(*map(np_to_torch, data)))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)),
                                          device=self.device, dtype=torch.bool)
            valid_next_state_batch = [
                s for s in batch.next_state if s is not None]
            non_final_next_states = None
            if len(valid_next_state_batch) > 0:
                non_final_next_states = torch.cat(
                    [s for s in batch.next_state if s is not None]).to(
                        self.device)

            state_batch = torch.cat(batch.state, 0).to(self.device)
            action_batch = torch.cat(batch.action, 0).to(self.device)

            state_action_values = self.acting_net(
                state_batch).gather(1, action_batch).to(self.device)

            true_state_values += state_action_values

            next_state_values = torch.zeros(batch_size, device=self.device)
            if non_final_next_states is not None:
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states).max(1)[0].detach()
            next_state_values = next_state_values.unsqueeze(1)
            true_next_state_values += next_state_values

        expected_state_action_values = (
            true_next_state_values * self.discount_factor) + reward_batch

        loss = self.loss_func(true_state_values,
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
        return loss.item()

    def record_transition(self, states, actions,
                          rewards, next_states, done):
        trans = self.process_transition(
            states, actions,
            rewards, next_states, done)
        self.buffer.store(trans)

    def update_policy(self):
        local_loss = {}
        for id_ in self.ids:
            buff = self.buffers[id_]
            batch_data = buff.sample(self.batch_size)
            policy_ = self.policies[id_]
            local_loss[id_] = policy_.learn_on_batch(
                batch_data)
        return local_loss

    def process_transition(self,
                           states, actions, rewards, next_states, done, info):
        processed_s = {}
        processed_a = {}
        processed_ns = {}
        processed_r = {
            "local": {},
            "central": np.reshape(rewards["central"], (1, 1)),
        }
        processed_t = {}
        for id_ in self.ids:
            processed_s[id_] = states[id_]
            processed_a[id_] = np.reshape(actions[id_], (1, 1))
            processed_r["local"][id_] = np.reshape(rewards[id_], (1, 1))
            processed_ns[id_] = next_states[id_]
            processed_t[id_] = np.array([[0 if done else 1]])
        return Transition(processed_s, processed_a,
                          processed_r, processed_ns, processed_t)
