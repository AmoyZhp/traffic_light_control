import logging
from typing import Dict, List
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from policy.core import Policy
from util.type import Transition
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class COMA(Policy):
    def __init__(self,
                 local_ids: List[str],
                 critic_net: nn.Module,
                 target_critic_net: nn.Module,
                 actor_nets: nn.Module,
                 learning_rate: float,
                 discount_factor: float,
                 device,
                 update_period: int,
                 action_space: int,
                 state_space: int,
                 local_obs_space: int) -> None:

        self.local_ids = local_ids
        self.ids_map = {}
        self.n_agents = len(self.local_ids)
        for i in range(self.n_agents):
            self.ids_map[self.local_ids[i]] = i
        self.device = device

        self.critic_net = critic_net
        self.target_critic_net = target_critic_net

        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        self.actor_nets = actor_nets

        self.critic_net.to(self.device)
        self.target_critic_net.to(self.device)
        for net_ in self.actor_nets.values():
            net_.to(self.device)

        self.critic_optim = optim.Adam(
            self.critic_net.parameters(), learning_rate)
        self.actor_optims = {}
        for id_, net_ in self.actor_nets.items():
            self.actor_optims[id_] = optim.Adam(
                net_.parameters(), learning_rate)

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.state_space = state_space
        self.local_obs_space = local_obs_space
        self.update_period = update_period
        self.update_count = 0
        self.critic_loss_func = nn.MSELoss()

    def compute_action(self, states, explore):
        actions = {}
        for id_, state in states["local"].items():
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            with torch.no_grad():
                value = self.actor_nets[id_](state)
                m = Categorical(value)
                action = m.sample()
                actions[id_] = action.item()
        return actions

    def learn_on_batch(self, batch_data: List[Transition]):
        batch_size = len(batch_data)
        if batch_size == 0:
            return 0.0

        for trans in batch_data:

            trans = self.__np_to_tensor(trans)

            critic_states = self.__cat_critic_state(trans)

            # traj_len * 1
            rewards: torch.Tensor = trans.reward["central"]
            joint_action: Dict[str, torch.Tensor] = trans.action

            joint_q_v, joint_selct_q_v, critic_loss = self.__critic_update(
                states=critic_states,
                actions=joint_action,
                rewards=rewards
            )

            _ = self.__actor_update(
                joint_obs=trans.state["local"],
                joint_action=joint_action,
                joint_q_v=joint_q_v,
                joint_selct_q_v=joint_selct_q_v,
            )

            return {
                "central": critic_loss,
                "local": {},
            }

    def __actor_update(self, joint_obs, joint_action,
                       joint_selct_q_v, joint_q_v):
        # n_agent * action_space
        actor_loss = {}
        for id_ in self.local_ids:
            local_obs = joint_obs[id_]
            action_prob = self.actor_nets[id_](
                local_obs)
            baseline = torch.sum(
                action_prob * joint_q_v[id_], dim=1).unsqueeze(-1)
            advantage = joint_selct_q_v[id_] - baseline
            log_prob = torch.log(action_prob).gather(
                1, joint_action[id_])
            # 负数的原因是因为算法默认是梯度下降，加了负号后就可以让它变成梯度上升
            actor_loss[id_] = -torch.sum(log_prob * advantage.detach())

        # 考虑到有 params share 的情况把更新放到最后
        for id_ in self.local_ids:

            self.actor_optims[id_].zero_grad()
            actor_loss[id_].backward()
            self.actor_optims[id_].step()

    def __critic_update(self,
                        states: Dict[str, torch.Tensor],
                        actions: Dict[str, torch.Tensor],
                        rewards: torch.Tensor):
        joint_selct_q_val = {}
        joint_q_val = {}
        expected_q_vals = {}
        for id_ in self.local_ids:
            ag_state = states[id_]
            ag_action = actions[id_]
            seq_len = ag_state.shape[0]
            next_selected_q_val = torch.zeros((seq_len, 1), device=self.device)
            next_selected_q_val[:-1] = self.target_critic_net(
                ag_state[1:]
            ).gather(1, ag_action[1:])

            expected_q_vals[id_] = next_selected_q_val.detach() * \
                self.discount_factor + rewards

        total_loss = 0.0
        for id_ in self.local_ids:
            ag_state = states[id_]
            ag_action = actions[id_]
            ag_exp_q_vals = expected_q_vals[id_]
            seq_q_val, seq_sel_q_val, loss = self.__local_critic_update(
                ag_state, ag_action, ag_exp_q_vals)
            joint_q_val[id_] = seq_q_val
            joint_selct_q_val[id_] = seq_sel_q_val
            total_loss += loss

        return joint_q_val, joint_selct_q_val, total_loss

    def __local_critic_update(self,
                              seq_state,
                              seq_action,
                              seq_exp_q_val):
        seq_q_val = []
        seq_sel_q_val = []
        seq_len = seq_state.shape[0]
        total_loss = 0.0
        for t in reversed(range(seq_len)):
            # action_space
            q_val = self.critic_net(seq_state[t]).unsqueeze(0)
            seq_q_val.append(q_val.detach())

            # 1 * 1
            selected_q_val = q_val.gather(
                1, seq_action[t].view(-1, 1)
            )
            seq_sel_q_val.append(selected_q_val.detach())

            critic_loss = self.critic_loss_func(
                q_val, seq_exp_q_val[t])
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            total_loss += critic_loss

            self.update_count += 1
            if self.update_count >= self.update_period:
                self.update_count = 0
                self.target_critic_net.load_state_dict(
                    self.critic_net.state_dict())

        seq_sel_q_val.reverse()
        seq_q_val.reverse()

        seq_q_val = torch.cat(seq_q_val, 0)
        seq_sel_q_val = torch.cat(seq_sel_q_val, 0)

        return seq_q_val, seq_sel_q_val, total_loss.item()

    def __cat_critic_state(self, batch: Transition) -> Dict[str, torch.Tensor]:
        joint_action_one_hot = []

        for id_ in self.local_ids:
            # seq * action_space
            action_one_hot = F.one_hot(batch.action[id_],
                                       self.action_space).squeeze(1)
            joint_action_one_hot.append(action_one_hot)

        # seq * (n * action_space)
        joint_action_one_hot = torch.cat(
            joint_action_one_hot, dim=1)

        joint_action_one_hot = joint_action_one_hot.unsqueeze(
            1).repeat(1, self.n_agents, 1)
        action_mask = (1 - torch.eye(self.n_agents))

        # n_agent * (n*action_space)
        action_mask = action_mask.view(-1, 1).repeat(
            1, self.action_space).view(self.n_agents, -1)

        # traj * n_agent * (n*action_space)
        action_mask = action_mask.unsqueeze(0).repeat(
            joint_action_one_hot.shape[0], 1, 1
        ).to(self.device)

        # traj * n_agent * (n * action_space)
        joint_action_one_hot = (
            joint_action_one_hot * action_mask).type(torch.float)
        # n_agent * seq * (n * action_space)
        joint_action_one_hot = joint_action_one_hot.permute(1, 0, 2)

        critic_states = {}
        central_state = batch.state["central"]
        for id_ in self.local_ids:
            local_obs = batch.state["local"][id_]
            agent_id_one_hot = F.one_hot(
                torch.tensor(self.ids_map[id_]), self.n_agents)

            agent_id_one_hot = agent_id_one_hot.view(1, -1).repeat(
                central_state.shape[0], 1).type(torch.float)
            critic_states[id_] = torch.cat(
                (central_state, local_obs,
                 agent_id_one_hot, joint_action_one_hot[self.ids_map[id_]]),
                dim=-1
            ).to(self.device)

        return critic_states

    def __np_to_tensor(self, trans: Transition) -> Transition:
        tensor_state = {
            "central": torch.tensor(
                trans.state["central"], dtype=torch.float,
                device=self.device,
            ),
            "local": {}
        }
        tensor_action = {}
        tensor_reward = {
            "central": torch.tensor(
                trans.reward["central"], dtype=torch.float,
                device=self.device
            ),
            "local": {}
        }
        tensor_next_state = {
            "central": torch.tensor(
                trans.next_state["central"], dtype=torch.float,
                device=self.device,
            ),
            "local": {}
        }
        tensor_done = torch.tensor(
            trans.done, dtype=torch.bool, device=self.device)
        for id_ in self.local_ids:
            tensor_state["local"][id_] = torch.tensor(
                trans.state["local"][id_],
                dtype=torch.float,
                device=self.device)
            tensor_action[id_] = torch.tensor(
                trans.action[id_],
                dtype=torch.long,
                device=self.device
            ).view(-1, 1)
            tensor_reward["local"][id_] = torch.tensor(
                trans.reward["local"][id_],
                dtype=torch.float,
                device=self.device,
            ).view(-1, 1)
            tensor_next_state["local"][id_] = torch.tensor(
                trans.next_state["local"][id_],
                dtype=torch.float,
                device=self.device,
            )
        batch = Transition(
            tensor_state, tensor_action, tensor_reward,
            tensor_next_state, tensor_done,
        )
        return batch

    def get_weight(self):
        actor_nets_weight = {}
        actor_optim_weight = {}
        for id_ in self.local_ids:
            actor_nets_weight[id_] = self.actor_nets[id_].state_dict()
            actor_optim_weight[id_] = self.actor_optims[id_].state_dict()
        weight = {
            "net": {
                "actors": actor_nets_weight,
                "critic": self.critic_net.state_dict(),
            },
            "optimizer": {
                "actors":  actor_optim_weight,
                "critic": self.critic_optim.state_dict(),
            }
        }
        return weight

    def set_weight(self, weight):
        net_w = weight["net"]
        optimizer_w = weight["optimizer"]
        for id_ in self.local_ids:
            self.actor_nets[id_].load_state_dict(
                net_w["actors"][id_])
            self.actor_optims[id_].load_state_dict(
                optimizer_w["actors"][id_])

        self.critic_net.load_state_dict(net_w["critic"])
        self.target_critic_net.load_state_dict(net_w["critic"])
        self.critic_optim.load_state_dict(optimizer_w["critic"])
