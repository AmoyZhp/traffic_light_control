from hprl.util.typing import Action, State, Transition, TransitionTuple
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hprl.policy.core import Policy


class VDN(Policy):
    def __init__(self,
                 agents_id: List[str],
                 acting_nets: Dict[str, nn.Module],
                 target_nets: Dict[str, nn.Module],
                 learning_rate: int,
                 discount_factor: float,
                 update_period: int,
                 action_space,
                 state_space,
                 device=None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.agents_id = agents_id
        params = []
        for id in self.agents_id:
            acting_nets[id].to(self.device)
            target_nets[id].to(self.device)

            target_nets[id].load_state_dict(acting_nets[id].state_dict())
            params.append({"params": acting_nets[id].parameters()})

        self.acting_nets = acting_nets
        self.target_nets = target_nets

        self.optimizer = optim.Adam(params, learning_rate)
        self.loss_func = nn.MSELoss()
        self.update_count = 0
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.update_period = update_period
        self.action_space = action_space
        self.state_space = state_space

    def compute_action(self, state: State) -> Action:
        actions = {}
        for id in self.agents_id:
            local_state = torch.tensor(
                state.local[id],
                dtype=torch.float,
                device=self.device)
            with torch.no_grad():
                value = self.acting_nets[id](local_state)
                value = np.squeeze(value, 0)
            _, index = torch.max(value, 0)
            actions[id] = index.item()
        return Action(local=actions)

    def learn_on_batch(self, batch_data: List[Transition]):
        if batch_data is None or not batch_data:
            return
        transition = self._to_tensor(batch_data)
        local_states = transition.state
        local_actions = transition.action
        local_next_state = transition.next_state
        terminal = transition.terminal
        central_reward = transition.reward
        q_val, next_q_val = self._cal_q_val(local_states, local_actions,
                                            local_next_state, terminal)
        print(q_val.shape)
        print(next_q_val.shape)
        expected_q_val = (next_q_val * self.discount_factor) + central_reward
        print(expected_q_val.shape)
        loss = self.loss_func(q_val, expected_q_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.update_period == 0:
            self.update_count = 0
            for id in self.acting_nets:
                self.target_nets[id].load_state_dict(
                    self.acting_nets[id].state_dict())

    def _to_tensor(self, batch_data: List[Transition]):
        states = {}
        actions = {}
        next_states = {}
        rewards = []
        terminal = []
        for id in self.agents_id:
            states[id] = []
            actions[id] = []
            next_states[id] = []
        for trans in batch_data:
            for id in self.agents_id:
                local_s = torch.tensor(trans.state.local[id],
                                       dtype=torch.float) .unsqueeze(0)
                states[id].append(local_s)

                local_a = torch.tensor(trans.action.local[id],
                                       dtype=torch.long).unsqueeze(0)
                actions[id].append(local_a)

                local_ns = torch.tensor(trans.next_state.local[id],
                                        dtype=torch.float).unsqueeze(0)
                next_states[id].append(local_ns)
                central_reward = torch.tensor(trans.reward.central,
                                              dtype=torch.float).view(-1, 1)
                rewards.append(central_reward)

                central_terminal = torch.tensor(trans.terminal.central,
                                                dtype=torch.long).view(-1, 1)
                terminal.append(central_terminal)
        states = torch.cat(states, 0).to(self.device)
        actions = torch.cat(actions, 0).to(self.device)
        next_states = torch.cat(next_states, 0).to(self.device)
        rewards = torch.cat(rewards, 0).to(self.device)
        terminal = torch.cat(terminal, 0).to(self.device)
        return TransitionTuple(states, actions, rewards, next_states, terminal)

    def _cal_q_val(self, states, actions, next_states, terminal):
        ns_mask = torch.tensor(tuple(map(lambda d: not d, terminal)),
                               device=self.device, dtype=torch.bool)
        central_q_val = 0.0
        central_next_q_val = 0.0
        for id in self.agents_id:
            q_val = self.acting_nets[id](
                states[id]
            ).gather(1, actions[id])
            next_q_val = torch.zeros_like(q_val)
            next_q_val[ns_mask] = self.target_nets[id](
                next_states[ns_mask]
            ).max(1)[0].detach().unqueeze(1)
            central_q_val += q_val
            central_next_q_val += next_q_val
        return central_q_val, central_next_q_val
