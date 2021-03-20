import logging
from hprl.util.typing import Action, MultiAgentSampleBatch, State, Transition, TransitionTuple
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hprl.policy.policy import MultiAgentPolicy

logger = logging.getLogger(__package__)


class QMIX(MultiAgentPolicy):
    def __init__(
        self,
        agents_id: List[str],
        actors_net: Dict[str, nn.Module],
        actors_target_net: Dict[str, nn.Module],
        critic_net: nn.Module,
        critic_target_net: nn.Module,
        critic_lr: int,
        discount_factor: float,
        update_period: int,
        action_space,
        state_space,
        loss_fn,
        prioritized,
        device=None,
    ):
        logger.info("QMIX init")
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.agents_id = agents_id

        self.actors_net = actors_net
        self.actors_target_net = actors_target_net
        self.critic_net = critic_net
        self.critic_target_net = critic_target_net
        self._update_target()

        self.critic_net.to(self.device)
        self.critic_target_net.to(self.device)
        params = list(self.critic_net.parameters())
        for id in self.agents_id:
            self.actors_net[id].to(self.device)
            self.actors_target_net[id].to(self.device)
            params += list(self.actors_net[id].parameters())
        self.optimizer = optim.Adam(params, critic_lr)
        self.loss_fn = loss_fn
        self.update_count = 0
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.update_period = update_period
        self.action_space = action_space
        self.state_space = state_space
        self.prioritized = prioritized
        logger.info("\t critic lr : %f", self.critic_lr)
        logger.info("\t discount factor : %f", self.discount_factor)
        logger.info("\t update period : %d", self.update_period)
        logger.info("\t prioritized : %s", self.prioritized)
        logger.info("\t loss fn : %s", self.loss_fn.__name__)
        for id in self.agents_id:
            action_space = self.action_space[id]
            state_space = self.state_space[id]
            logger.info("\t agents %s", id)
            logger.info("\t\t action space is %s", action_space)
            logger.info("\t\t state space is %s", state_space)
        logger.info("QMIX init done")

    def compute_action(self, state: State) -> Action:
        actions = {}
        for id in self.agents_id:
            local_state = torch.tensor(
                state.local[id],
                dtype=torch.float,
                device=self.device,
            )
            with torch.no_grad():
                value = self.actors_net[id](local_state)
                value = np.squeeze(value, 0)
            _, index = torch.max(value, 0)
            actions[id] = index.item()
        return Action(local=actions)

    def learn_on_batch(self, sample_batch: MultiAgentSampleBatch):
        if sample_batch is None:
            return {}
        trans = sample_batch.transitions
        if not trans:
            return {}
        transition, central_s, central_next_s = self._to_tensor(trans)
        local_states = transition.state
        local_actions = transition.action
        local_next_state = transition.next_state
        terminal = transition.terminal
        central_reward = transition.reward
        agents_q, agents_next_q = self._cal_agents_q(
            local_states,
            local_actions,
            local_next_state,
            terminal,
        )
        q_val = self.critic_net(agents_q, central_s)
        next_q_val = self.critic_target_net(agents_next_q, central_next_s)

        expected_q_val = (next_q_val * self.discount_factor) + central_reward
        loss = 0.0
        info = {}
        if self.prioritized:
            weights = torch.tensor(sample_batch.weigths, dtype=torch.float)
            weights = weights.unsqueeze(-1).to(self.device)
            loss = self.loss_fn(
                q_val,
                expected_q_val.detach(),
                weights.detach(),
            )
            td_error = (expected_q_val - q_val).squeeze(-1)
            priorities = torch.clamp(td_error, -1, 1).detach()
            priorities = abs(priorities) + 1e-6
            priorities = priorities.tolist()
            info["priorities"] = priorities
        else:
            loss = self.loss_fn(q_val, expected_q_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.update_period == 0:
            self.update_count = 0
            self._update_target()
        return info

    def _to_tensor(self, batch_data: List[Transition]):
        central_states = []
        central_next_states = []
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
            central_s = torch.tensor(trans.state.central,
                                     dtype=torch.float).unsqueeze(0)
            central_states.append(central_s)

            central_next_s = torch.tensor(trans.next_state.central,
                                          dtype=torch.float).unsqueeze(0)
            central_next_states.append(central_next_s)

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
        central_states = torch.cat(central_states, 0).to(self.device)
        central_next_states = torch.cat(central_next_states, 0).to(self.device)
        local_tuple = TransitionTuple(
            states,
            actions,
            rewards,
            next_states,
            terminal,
        )
        return local_tuple, central_states, central_next_states

    def _cal_agents_q(self, states, actions, next_states, terminal):
        agents_q = []
        agents_next_q = []
        ns_mask = torch.tensor(
            tuple(map(lambda d: not d, terminal)),
            device=self.device,
            dtype=torch.bool,
        )
        for id in self.agents_id:
            chosen_q = self.actors_net[id](states[id]).gather(1, actions[id])
            chosen_next_q = torch.zeros_like(chosen_q)
            chosen_next_q[ns_mask] = self.actors_target_net[id](
                next_states[id][ns_mask]).max(1)[0].unsqueeze(1)

            agents_q.append(chosen_q.view(-1, 1, 1))
            agents_next_q.append(chosen_next_q.view(-1, 1, 1))
        agents_q = torch.cat(agents_q, 1).to(self.device)
        agents_next_q = torch.cat(agents_next_q, 1).to(self.device)
        return agents_q, agents_next_q

    def _update_target(self):
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        for id in self.agents_id:
            self.actors_target_net[id].load_state_dict(
                self.actors_net[id].state_dict())

    def get_config(self):
        config = {
            "critic_lr": self.critic_lr,
            "discount_factor": self.discount_factor,
            "update_period": self.update_period,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }
        return config

    def get_weight(self):
        local_weight = {}
        for id_ in self.agents_id:
            local_weight[id_] = self.actors_net[id_].state_dict()
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
            self.actors_net[id_].load_state_dict(net_w[id_])
            self.actors_target_net[id_].load_state_dict(net_w[id_])
