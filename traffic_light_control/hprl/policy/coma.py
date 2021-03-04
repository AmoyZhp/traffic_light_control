from hprl.policy.samples_processor import parase_traj_list, parase_trajectory_to_tensor
from hprl.util.typing import Action, State, Trajectory, TrajectoryTuple
from typing import Dict, List
from hprl.policy.core import Policy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class COMA(Policy):
    def __init__(self,
                 agents_id: List[str],
                 critic_net: nn.Module,
                 target_critic_net: nn.Module,
                 actors_net: Dict[str, nn.Module],
                 learning_rate: float,
                 discount_factor: float,
                 update_period: int,
                 clip_param: float,
                 inner_epoch: int,
                 local_action_space,
                 local_state_space,
                 device=None) -> None:

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.agents_id = agents_id
        self.ids_map: Dict[str, int] = {}
        self.agents_count = len(self.agents_id)
        for i in range(self.agents_count):
            self.ids_map[self.agents_id[i]] = i

        self.critic_net = critic_net
        self.target_critic_net = target_critic_net
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.critic_net.to(self.device)
        self.target_critic_net.to(self.device)

        self.actors_net = actors_net
        for net in self.actors_net.values():
            net.to(self.device)

        self.critic_optim = optim.Adam(self.critic_net.parameters(),
                                       learning_rate)
        self.actors_optim = {}
        for id in self.agents_id:
            self.actors_optim[id] = optim.Adam(
                self.actors_net[id].parameters(), learning_rate)

        self.discount_factor = discount_factor
        self.update_period = update_period
        self.learning_rate = learning_rate
        self.clip_param = clip_param
        self.inner_epoch = inner_epoch
        self.local_s_space = local_state_space
        self.local_a_space = local_action_space
        self.update_count = 0

    def compute_action(self, state: State) -> Action:
        actions = {}
        for id in self.agents_id:
            local_s = torch.tensor(
                state.local[id],
                dtype=torch.float,
                device=self.device,
            )
            with torch.no_grad():
                value = self.actors_net[id](local_s)
                m = Categorical(value)
                actions[id] = m.sample().item()
        return Action(local=actions)

    def learn_on_batch(self, batch_data: List[Trajectory]):
        if batch_data is None or not batch_data:
            return

        batch_seq_data = parase_traj_list(batch_data)

        central_s = batch_seq_data.states["central"].to(self.device)
        agents_state = batch_seq_data.states["local"]
        agents_action = batch_seq_data.actions["local"]
        for id in self.agents_id:
            agents_state[id] = agents_state[id].to(self.device)
            agents_action[id] = agents_action[id].to(self.device)

        logger.debug("agent local state shape is {}".format(
            agents_state[self.agents_id[0]].shape))
        logger.debug("agent action shape is {}".format(
            agents_action[self.agents_id[0]].shape))
        logger.debug("agent central state shape is {}".format(central_s.shape))

        critic_states = self._cat_critic_states(
            central_s=central_s,
            agents_states=agents_state,
            agents_action=agents_action,
        )
        central_reward = batch_seq_data.rewards["central"].to(self.device)

        selected_old_a_probs = {}
        for id in self.agents_id:
            a_prob = self.actors_net[id](agents_state[id])
            selected_old_a_probs[id] = a_prob.gather(-1, agents_action[id]).to(
                self.device).detach()

        logger.debug("critic states shape is {}".format(
            critic_states[self.agents_id[0]].shape))
        logger.debug("central reward shape is {}".format(central_reward.shape))
        logger.debug("old action probs shape is {}".format(
            selected_old_a_probs[self.agents_id[0]].shape))

        for _ in range(self.inner_epoch):
            critic_loss, actor_losses = self._inner_loop(
                critic_states=critic_states,
                agents_state=agents_state,
                agents_action=agents_action,
                central_reward=central_reward,
                selected_old_a_probs=selected_old_a_probs,
            )

            for id in self.agents_id:
                self.actors_optim[id].zero_grad()
                actor_losses[id].backward()
                self.actors_optim[id].step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            self.update_count += 1
            if self.update_count > self.update_period:
                self.update_count = 0
                self.target_critic_net.load_state_dict(
                    self.critic_net.state_dict())

    def _inner_loop(self, critic_states: Dict[str, torch.Tensor],
                    agents_state: Dict[str, torch.Tensor],
                    agents_action: Dict[str, torch.Tensor],
                    central_reward: torch.Tensor,
                    selected_old_a_probs: Dict[str, torch.Tensor]):
        q_vals, selected_q_vals, critic_loss = self._compute_critic_loss(
            states=critic_states,
            actions=agents_action,
            rewards=central_reward,
        )

        logger.debug(f"q val shape {q_vals[self.agents_id[0]].shape}")
        logger.debug(
            f"selected q val shape {selected_q_vals[self.agents_id[0]].shape}")

        actor_losses = self._compute_actor_loss(
            agents_state=agents_state,
            agents_action=agents_action,
            q_vals=q_vals,
            selected_q_vals=selected_q_vals,
            selected_old_a_probs=selected_old_a_probs,
        )

        return critic_loss, actor_losses

    def _compute_critic_loss(self, states: Dict[str, torch.Tensor],
                             actions: Dict[str, torch.Tensor],
                             rewards: torch.Tensor):
        q_vals = {}
        selected_q_vals = {}
        critic_loss = 0.0
        for id in self.agents_id:
            # its not like local state
            # each agent has its version of critic state
            agent_s = states[id]
            local_a = actions[id]

            q_val = self.critic_net(agent_s)
            selected_q_val = q_val.gather(-1, local_a)

            next_selected_q_v = torch.zeros_like(q_val, device=self.device)
            next_selected_q_v[:-1] = self.target_critic_net(
                agent_s[1:]).gather(-1, local_a[1:])

            expected_q_v = (next_selected_q_v * self.discount_factor + rewards)
            loss = self._critic_loss_func(selected_q_val,
                                          expected_q_v.detach())

            critic_loss += loss
            q_vals[id] = q_val
            selected_q_vals[id] = selected_q_val
        critic_loss /= self.agents_count
        return q_vals, selected_q_vals, critic_loss

    def _compute_actor_loss(self, agents_state: Dict[str, torch.tensor],
                            agents_action: Dict[str, torch.tensor],
                            q_vals: Dict[str, torch.tensor],
                            selected_q_vals: Dict[str, torch.tensor],
                            selected_old_a_probs: Dict[str, torch.tensor]):
        actors_loss = {}
        for id in self.agents_id:
            local_s = agents_state[id]
            local_a = agents_action[id]

            action_prob = self.actors_net[id](local_s)
            selected_a_prob = action_prob.gather(-1, local_a)

            baseline = torch.sum(action_prob * q_vals[id],
                                 dim=-1).unsqueeze(-1)

            advantage = selected_q_vals[id] - baseline
            advantage = advantage.detach()
            logger.debug("advantage shape {}".format(advantage.shape))

            ratio = selected_a_prob / selected_old_a_probs[id]
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param,
                                1 + self.clip_param) * advantage
            actors_loss[id] = -torch.min(surr1, surr2).mean()
        return actors_loss

    def _cat_critic_states(self, central_s: torch.Tensor,
                           agents_states: Dict[str, torch.Tensor],
                           agents_action: Dict[str, torch.Tensor]):

        joint_action_one_hot = self._compute_joint_action_one_hot(
            agents_action)
        logger.debug("joint action one hot shape is {}".format(
            joint_action_one_hot.shape))
        critic_states = {}
        batch_size = central_s.shape[0]
        seq_len = central_s.shape[1]
        for id in self.agents_id:
            local_s = agents_states[id]
            agent_id_one_hot = F.one_hot(torch.tensor(self.ids_map[id]),
                                         self.agents_count)
            agent_id_one_hot = agent_id_one_hot.view(1, -1).repeat(
                batch_size, seq_len, 1).type(torch.float).to(self.device)

            critic_states[id] = torch.cat(
                (central_s, local_s, agent_id_one_hot,
                 joint_action_one_hot[self.ids_map[id]]),
                dim=-1).to(self.device)
        return critic_states

    def _compute_joint_action_one_hot(self, agents_action):
        joint_action_one_hot = []
        for id in self.agents_id:
            action_one_hot = F.one_hot(agents_action[id],
                                       self.local_a_space).squeeze(-2)
            joint_action_one_hot.append(action_one_hot)
        joint_action_one_hot = torch.cat(joint_action_one_hot, dim=-1)

        batch_size = joint_action_one_hot.shape[0]
        seq_len = joint_action_one_hot.shape[1]
        joint_action_one_hot = joint_action_one_hot.unsqueeze(2).repeat(
            1, 1, self.agents_count, 1).to(self.device)
        action_mask = (1 - torch.eye(self.agents_count)).view(-1, 1)
        action_mask = action_mask.repeat(1, self.local_a_space).view(
            self.agents_count, -1)
        # traj * n_agent * (n*action_space)

        action_mask = action_mask.unsqueeze(0).unsqueeze(0).repeat(
            batch_size, seq_len, 1, 1).to(self.device)

        # traj * n_agent * (n * action_space)
        joint_action_one_hot = (joint_action_one_hot * action_mask).type(
            torch.float)
        joint_action_one_hot = joint_action_one_hot.permute(2, 0, 1, 3)
        joint_action_one_hot = joint_action_one_hot.to(self.device)
        return joint_action_one_hot

    def get_weight(self):
        actor_nets_weight = {}
        actor_optim_weight = {}
        for id_ in self.agents_id:
            actor_nets_weight[id_] = self.actors_net[id_].state_dict()
            actor_optim_weight[id_] = self.actors_optim[id_].state_dict()
        weight = {
            "net": {
                "actors": actor_nets_weight,
                "critic": self.critic_net.state_dict(),
            },
            "optimizer": {
                "actors": actor_optim_weight,
                "critic": self.critic_optim.state_dict(),
            }
        }
        return weight

    def set_weight(self, weight):
        net_w = weight["net"]
        optimizer_w = weight["optimizer"]
        for id_ in self.agents_id:
            self.actors_net[id_].load_state_dict(net_w["actors"][id_])
            self.actors_optim[id_].load_state_dict(optimizer_w["actors"][id_])

        self.critic_net.load_state_dict(net_w["critic"])
        self.target_critic_net.load_state_dict(net_w["critic"])
        self.critic_optim.load_state_dict(optimizer_w["critic"])

    def get_config(self):
        config = {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "update_period": self.update_period,
            "local_action_space": self.local_a_space,
            "local_state_space": self.local_s_space,
            "clip_param": self.clip_param,
            "inner_epoch": self.inner_epoch,
        }
        return config

    def unwrapped(self):
        return self

    def _critic_loss_func(self, input, target):
        return torch.mean((input - target)**2)
