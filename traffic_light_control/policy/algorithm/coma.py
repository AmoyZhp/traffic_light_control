import logging
from typing import Dict, List
from torch import Tensor, optim
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

        seq_batch_data = self.__cat_batch_data(batch_data)

        critic_states = self.__cat_critic_state(
            agents_actions=seq_batch_data.action,
            central_state=seq_batch_data.state["central"],
            agents_local_obs=seq_batch_data.state["local"]
        )

        agents_q_v, agent_sel_q_v, agents_critic_loss = self.__cal_critic_loss(
            states=critic_states,
            actions=seq_batch_data.action,
            rewards=seq_batch_data.reward["central"],
        )

        agents_actor_loss = self.__cal_actor_loss(
            agents_local_obs=seq_batch_data.state["local"],
            agents_actions=seq_batch_data.action,
            agents_q_v=agents_q_v,
            agents_selct_q_v=agent_sel_q_v,
        )

        critic_loss = 0.0
        for id_ in self.local_ids:
            critic_loss += agents_critic_loss[id_]
            self.critic_optim.zero_grad()
            agents_critic_loss[id_].backward()
            self.critic_optim.step()

            self.actor_optims[id_].zero_grad()
            agents_actor_loss[id_].backward()
            self.actor_optims[id_].step()

        return {
            "central": critic_loss.item(),
            "local": {},
        }

    def __cal_actor_loss(self,
                         agents_local_obs, agents_actions,
                         agents_selct_q_v, agents_q_v):
        # n_agent * action_space
        actor_loss = {}
        for id_ in self.local_ids:
            local_obs = agents_local_obs[id_]
            action_prob = self.actor_nets[id_](
                local_obs)
            baseline = torch.sum(
                action_prob * agents_q_v[id_], dim=1).unsqueeze(-1)
            print("baseline shape {}".format(baseline.shape))
            advantage = agents_selct_q_v[id_] - baseline
            print("adv shape {}".format(advantage.shape))
            log_prob = torch.log(action_prob).gather(
                -1, agents_actions[id_])
            # 负数的原因是因为算法默认是梯度下降，加了负号后就可以让它变成梯度上升
            actor_loss[id_] = -torch.sum(log_prob * advantage.detach())

        return actor_loss

    def __cal_critic_loss(self,
                          states: Dict[str, torch.Tensor],
                          actions: Dict[str, torch.Tensor],
                          rewards: torch.Tensor):
        expected_q_vals = {}
        for id_ in self.local_ids:
            ag_state = states[id_]
            ag_action = actions[id_]

        agents_q_val = {}
        agents_sel_q_val = {}
        agents_local_loss = {}
        for id_ in self.local_ids:
            ag_state = states[id_]
            ag_action = actions[id_]
            seq_len = ag_state.shape[0]
            batch_size = ag_state.shape[1]

            q_vals = self.critic_net(ag_state)
            sel_q_vals = q_vals.gather(
                -1, ag_action
            )

            next_sel_q_val = torch.zeros(
                (seq_len, batch_size, 1), device=self.device)
            next_sel_q_val[:-1] = self.target_critic_net(
                ag_state[1:]
            ).gather(-1, ag_action[1:])

            expected_q_vals = next_sel_q_val * \
                self.discount_factor + rewards

            loss = self.__critic_loss_func(
                sel_q_vals, expected_q_vals.detach())
            agents_q_val[id_] = q_vals
            agents_sel_q_val[id_] = sel_q_vals
            agents_local_loss[id_] = loss

        return agents_q_val, agents_sel_q_val, agents_local_loss

    def __cat_critic_state(self,
                           agents_actions: Dict[str, Tensor],
                           central_state: Tensor,
                           agents_local_obs: Dict[str, Tensor]
                           ) -> Dict[str, torch.Tensor]:
        joint_action_one_hot = []

        for id_ in self.local_ids:
            # seq * action_space
            action_one_hot = F.one_hot(agents_actions[id_],
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
        for id_ in self.local_ids:
            local_obs = agents_local_obs[id_]
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

    def __cat_batch_data(self, batch_data) -> List[Transition]:

        def np_to_tensor(trans: Transition) -> Transition:
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

        batch_data = list(map(batch_data, np_to_tensor))
        central_states = []
        central_rewards = []
        agents_actions = {}
        agents_local_obs = {}
        for id_ in self.local_ids:
            agents_actions[id_] = []
            agents_local_obs[id_] = []

        for trans in batch_data:
            central_states.append(trans.state["central"].unsqueeze(1))
            central_rewards.append(trans.reward["central"].unsqueeze(1))
            for k, v in trans.state["local"].items():
                agents_local_obs[k].append(v.unsqueeze(1))
            for k, v in trans.action.items():
                agents_actions[k].append(v.unsqueeze(1))
        central_states = torch.cat(central_states, dim=1)
        central_rewards = torch.cat(central_rewards, dim=1)
        for id_ in self.local_ids:
            agents_actions[id_] = torch.cat(agents_actions[id_], dim=1)
            agents_local_obs[id_] = torch.cat(agents_local_obs[id_], dim=1)

        seq_batch_data = Transition(
            {
                "central": central_states,
                "local": agents_local_obs,
            },
            agents_actions,
            central_rewards,
            {},
            {}
        )
        return seq_batch_data

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

    def __critic_loss_func(self, input, target):
        return torch.mean((input - target) ** 2)
