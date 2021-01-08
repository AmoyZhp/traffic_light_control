from torch import batch_norm, device, optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from policy.core import Policy
from util.type import Transition
from torch.distributions import Categorical


class COMA(Policy):
    def __init__(self,
                 local_ids,
                 critic_net,
                 actor_nets,
                 learning_rate,
                 discount_factor,
                 device,
                 action_space,
                 state_space,
                 local_obs_space) -> None:

        self.local_ids = local_ids
        self.ids_map = {}
        self.n_agents = len(self.local_ids)
        for i in range(self.n_agents):
            self.ids_map[self.local_ids[i]] = i
        self.device = device
        self.critic_net = critic_net
        self.actor_nets = actor_nets

        self.critic_net.to(self.device)
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

    def learn_on_batch(self, batch_data):
        batch_size = len(batch_data)
        if batch_size == 0:
            return 0.0
        actor_loss = 0.0
        critic_loss = 0.0
        for trans in batch_data:

            batch = self.__process_tran(trans)
            # traj_len * n_agents * state_space
            central_state_n_a = batch.state["central"].unsqueeze(1).repeat(
                1, self.n_agents, 1)
            # print("central state shape {}".format(central_state_n_a.shape))
            # print("central_state {}".format(central_state_n_a))

            joint_action_one_hot = []
            joint_action_raw = []
            local_obs_cat = []
            agent_ids_cat = []

            for id_ in self.local_ids:
                # tranj * (n*action_space)
                action_one_hot = F.one_hot(batch.action[id_],
                                           self.action_space)
                joint_action_one_hot.append(action_one_hot)

                joint_action_raw.append(
                    batch.action[id_].view(-1, 1))

                local_obs_cat.append(batch.state["local"][id_].unsqueeze(1))

                # 1 * n_agent
                agent_id_one_hot = F.one_hot(
                    torch.tensor(self.ids_map[id_]), self.n_agents)
                agent_ids_cat.append(agent_id_one_hot.view(1, -1))

            # tranj * n_a * obs_space
            local_obs_cat = torch.cat(local_obs_cat, dim=1)
            # print("local obs shape {}".format(local_obs_cat.shape))
            # print("local_obs {}".format(local_obs_cat))

            # traj * (n * action_space)
            joint_action_one_hot = torch.cat(
                joint_action_one_hot, dim=1)

            # traj * n_agent * (n * action_space)
            joint_action_one_hot = joint_action_one_hot.unsqueeze(
                1).repeat(1, self.n_agents, 1)
            # print("actions one hot cat repeate shape {}".format(
            #     joint_action_one_hot.shape))
            # print("actions one hot cat repeate {}".format(
            #     joint_action_one_hot))
            action_mask = (1 - torch.eye(self.n_agents))
            # n_agent * (n*action_space)
            action_mask = action_mask.view(-1, 1).repeat(
                1, self.action_space).view(self.n_agents, -1)

            # traj * n_agent * (n*action_space)
            action_mask = action_mask.unsqueeze(0).repeat(
                joint_action_one_hot.shape[0], 1, 1
            )

            # traj * n_agent * (n * action_space)
            joint_action_one_hot = (
                joint_action_one_hot * action_mask).type(torch.float)
            # print("actions one hot cat mask shape {}".format(
            #     joint_action_one_hot.shape))
            # print("actions one hot cat mask  {}".format(
            #     joint_action_one_hot))

            # n_a * n_a
            agent_ids_cat = torch.cat(agent_ids_cat, dim=0)
            agent_ids_cat = agent_ids_cat.unsqueeze(
                0).repeat(central_state_n_a.shape[0], 1, 1).type(torch.float)
            # print("agent ids shape {} ".format(agent_ids_cat.shape))
            # print("agent ids {} ".format(agent_ids_cat))

            # traj_len * n_agent * criti_state_space
            critic_states = torch.cat(
                (central_state_n_a, local_obs_cat,
                 agent_ids_cat, joint_action_one_hot),
                dim=-1
            )
            # print("critic state shape {}".format(critic_states.shape))
            # print("criti state {}".format(critic_states))

            # traj_len * n_agent * 1
            rewards = batch.reward["central"].unsqueeze(
                1).repeat(1, self.n_agents, 1)

            # traj_len * n_agent
            joint_action_raw = torch.cat(joint_action_raw, dim=1)
            # print("joint action raw shape {}".format(
            #     joint_action_raw.shape
            # ))
            # print("joint action raw {}".format(
            #     joint_action_raw
            # ))

            tranj_selected_q_v = []
            tranj_q_v = []

            for t in reversed(range(critic_states.shape[0])):
                # n_agent * action_space
                q_values = self.critic_net(critic_states[t])
                tranj_q_v.append(q_values.unsqueeze(0))

                # n_agent * 1
                # print("q_values {}".format(q_values))
                selected_q_values = q_values.gather(
                    1, joint_action_raw[t].view(-1, 1)
                )
                # print("selected q values {}".format(selected_q_values))
                tranj_selected_q_v.append(selected_q_values.unsqueeze(0))

                selected_next_q_values = torch.zeros_like(selected_q_values)
                if t+1 < critic_states.shape[0]:
                    next_q_values = self.critic_net(critic_states[t+1])
                    selected_next_q_values = next_q_values.gather(
                        1, joint_action_raw[t+1].view(-1, 1)
                    )
                excepted_q_values = selected_next_q_values * \
                    self.discount_factor + rewards[t]

                critic_loss = self.critic_loss_func(
                    selected_q_values, excepted_q_values)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            tranj_selected_q_v.reverse()
            tranj_q_v.reverse()
            tranj_selected_q_v = torch.cat(tranj_selected_q_v, dim=0)
            tranj_q_v = torch.cat(tranj_q_v, dim=0)

            for t in reversed(range(local_obs_cat.shape[0])):
                # n_agent * action_space
                action_prob = self.actor_nets[self.local_ids[0]](
                    local_obs_cat[t])
                # print("action prob {}".format(
                #     action_prob
                # ))
                # print("selected q values {}".format(
                #     tranj_selected_q_v[t]))
                # print("q values {}".format(
                #     tranj_q_v[t]))
                baseline = torch.sum(
                    action_prob * tranj_q_v[t], dim=1).view(-1, 1)
                # print("baseline {}".format(baseline))
                advantage = tranj_selected_q_v[t] - baseline
                # print("advantage {}".format(advantage))
                m = Categorical(action_prob)
                log_prob = m.log_prob(joint_action_raw[t])
                # 负数的原因是因为算法默认是梯度下降，加了负号后就可以让它变成梯度上升
                actor_loss = -torch.sum(log_prob * advantage.detach())
                self.actor_optims[self.local_ids[0]].zero_grad()
                actor_loss.backward()
                self.actor_optims[self.local_ids[0]].step()
            return {
                "central": critic_loss.item(),
                "local": {}
            }

    def __compute_reward_to_go(self, rewards):
        rewards = rewards.view(-1, 1)
        weight = torch.triu(torch.ones((rewards.shape[0], rewards.shape[0]),
                                       device=self.device))
        rtg = weight.matmul(rewards)

        return rtg

    def __process_tran(self, trans):
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
            )
            tensor_reward["local"][id_] = torch.tensor(
                trans.reward["local"][id_],
                dtype=torch.float,
                device=self.device,
            )
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
        self.critic_optim.load_state_dict(optimizer_w["critic"])
