import logging
from hprl.util.typing import Action, MultiAgentSampleBatch, State, Transition, TransitionTuple
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hprl.policy.policy import MultiAgentPolicy

logger = logging.getLogger(__package__)


class VDN(MultiAgentPolicy):
    def __init__(
        self,
        agents_id: List[str],
        acting_nets: Dict[str, nn.Module],
        target_nets: Dict[str, nn.Module],
        critic_lr: int,
        discount_factor: float,
        update_period: int,
        local_action_space,
        local_state_space,
        loss_fn,
        prioritized,
        device=None,
    ):
        logger.info("VDN init")
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

        self.optimizer = optim.Adam(params, critic_lr)
        self.loss_fn = loss_fn
        self.update_count = 0
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.update_period = update_period
        self.local_action_space = local_action_space
        self.local_state_space = local_state_space
        self.prioritized = prioritized
        logger.info("\t critic lr : %f", self.critic_lr)
        logger.info("\t discount factor : %f", self.discount_factor)
        logger.info("\t update period : %d", self.update_period)
        logger.info("\t prioritized : %s", self.prioritized)
        for id in self.agents_id:
            local_action_space = self.local_action_space[id]
            local_state_space = self.local_state_space[id]
            logger.info("\t agents %s", id)
            logger.info("\t\t action space is %s", local_action_space)
            logger.info("\t\t state space is %s", local_state_space)
        logger.info("VDN init done")

    def compute_action(self, state: State) -> Action:
        actions = {}
        for id in self.agents_id:
            local_state = torch.tensor(
                state.local[id],
                dtype=torch.float,
                device=self.device,
            )
            with torch.no_grad():
                value = self.acting_nets[id](local_state)
                value = np.squeeze(value, 0)
            _, index = torch.max(value, 0)
            actions[id] = index.item()
        return Action(local=actions)

    def learn_on_batch(self, sample_batch: MultiAgentSampleBatch):
        if sample_batch is None:
            return {}
        trans_batch = sample_batch.transitions
        if not trans_batch:
            return {}
        transition = self._to_tensor(trans_batch)
        local_states = transition.state
        local_actions = transition.action
        local_next_state = transition.next_state
        terminal = transition.terminal
        central_reward = transition.reward
        q_vals, next_q_val = self._cal_q_val(
            local_states,
            local_actions,
            local_next_state,
            terminal,
        )
        expected_q_vals = (next_q_val * self.discount_factor) + central_reward
        loss = 0.0
        info = {}
        if self.prioritized:
            weights = torch.tensor(sample_batch.weigths, dtype=torch.float)
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
            self.update_count = 0
            for id in self.acting_nets:
                self.target_nets[id].load_state_dict(
                    self.acting_nets[id].state_dict())
        return info

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
                                       dtype=torch.float).unsqueeze(0)
                states[id].append(local_s)

                local_a = torch.tensor(trans.action.local[id],
                                       dtype=torch.long).view(-1, 1)
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
        for id in self.agents_id:
            states[id] = torch.cat(states[id], 0).to(self.device)
            actions[id] = torch.cat(actions[id], 0).to(self.device)
            next_states[id] = torch.cat(next_states[id], 0).to(self.device)
        rewards = torch.cat(rewards, 0).to(self.device)
        terminal = torch.cat(terminal, 0).to(self.device)
        return TransitionTuple(states, actions, rewards, next_states, terminal)

    def _cal_q_val(self, states, actions, next_states, terminal):
        ns_mask = torch.tensor(tuple(map(lambda d: not d, terminal)),
                               device=self.device,
                               dtype=torch.bool)
        central_q_val = 0.0
        central_next_q_val = 0.0
        for id in self.agents_id:
            q_val = self.acting_nets[id](states[id]).gather(1, actions[id])
            next_q_val = torch.zeros_like(q_val)
            next_q_val[ns_mask] = self.target_nets[id](
                next_states[id][ns_mask]).max(1)[0].detach().unsqueeze(1)
            central_q_val += q_val
            central_next_q_val += next_q_val
        return central_q_val, central_next_q_val

    def get_config(self):
        config = {
            "critic_lr": self.critic_lr,
            "discount_factor": self.discount_factor,
            "update_period": self.update_period,
            "local_action_space": self.local_action_space,
            "local_state_space": self.local_state_space,
        }
        return config

    def get_weight(self):
        local_weight = {}
        for id_ in self.agents_id:
            local_weight[id_] = self.acting_nets[id_].state_dict()
        weight = {
            "net": local_weight,
            "optimizer": self.optimizer.state_dict(),
        }
        return weight

    def unwrapped(self):
        return self

    def set_weight(self, weight):
        optimizer_w = weight["optimizer"]
        self.optimizer.load_state_dict(optimizer_w)
        self.step = weight["step"]

        net_w = weight["net"]
        for id_ in self.agents_id:
            self.acting_nets[id_].load_state_dict(net_w[id_])
            self.target_nets[id_].load_state_dict(net_w[id_])
