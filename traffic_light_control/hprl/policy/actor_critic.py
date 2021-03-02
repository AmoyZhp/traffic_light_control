from hprl.policy.nets import CartPole, CartPolePG
from hprl.util.enum import AdvantageTypes, TrainnerTypes, ReplayBufferTypes
from hprl.replaybuffer.core import ReplayBuffer
from hprl.util.typing import Action, State, Trajectory
from typing import List
from torch import optim
import torch
import torch.nn as nn
from hprl.policy.core import Policy
from hprl.policy.core import to_tensor_for_trajectory, compute_reward_to_go
from torch.distributions import Categorical


def get_ac_default_config():
    capacity = 4000
    learning_rate = 1e-3
    batch_size = 16
    discount_factor = 0.99
    update_period = 50
    action_space = 2
    state_space = 4

    policy_config = {
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "update_period": update_period,
        "action_space": action_space,
        "state_space": state_space,
    }
    buffer_config = {
        "type": ReplayBufferTypes.Common,
        "capacity": capacity,
    }
    exec_config = {
        "batch_size": batch_size,
        "base_dir": "records",
        "check_frequency": 100,
    }
    trainner_config = {
        "type": TrainnerTypes.IAC,
        "executing": exec_config,
        "policy": policy_config,
        "buffer": buffer_config,
    }

    critic_net = CartPole(
        input_space=state_space,
        output_space=action_space
    )

    critic_target_net = CartPole(
        input_space=state_space,
        output_space=action_space
    )

    actor_net = CartPolePG(
        input_space=state_space,
        output_space=action_space,
    )

    model = {
        "critic_net": critic_net,
        "critic_target_net": critic_target_net,
        "actor_net": actor_net
    }

    return trainner_config, model


class ActorCritic(Policy):
    def __init__(self,
                 critic_net: nn.Module,
                 critic_target_net: nn.Module,
                 actor_net: nn.Module,
                 learning_rate: float,
                 discount_factor: float,
                 update_period: int,
                 action_space,
                 state_space,
                 device: torch.device = None,
                 advantage_type: AdvantageTypes = AdvantageTypes.RewardToGO) -> None:
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.actor_net = actor_net
        self.critic_net = critic_net
        self.critic_target_net = critic_target_net

        self.critic_target_net.load_state_dict(self.critic_net.state_dict())

        self.critic_net.to(self.device)
        self.critic_target_net.to(self.device)
        self.actor_net.to(self.device)

        self.critic_optim = optim.Adam(
            self.critic_net.parameters(), learning_rate)
        self.actor_optim = optim.Adam(
            self.actor_net.parameters(), learning_rate)

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.state_space = state_space
        self.update_count = 0
        self.update_period = update_period
        self.advantage_type = advantage_type

    def compute_action(self, state: State) -> Action:
        state = torch.tensor(
            state.central, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.actor_net(state)
            m = Categorical(value)
            action = m.sample()
            return Action(central=action.item())

    def learn_on_batch(self, batch_data: List[Trajectory]):
        batch_size = len(batch_data)
        if batch_size == 0:
            return 0.0
        traj_len = len(batch_data[0].states)
        traj_len_equal = True
        for traj in batch_data:
            if traj_len != len(traj.states):
                traj_len_equal = False
                break

        states, actions, rewards = to_tensor_for_trajectory(
            batch_data,
            self.device)
        critic_loss = 0.0
        actor_loss = 0.0
        if traj_len_equal:
            # if they have equal sequence length
            # they could be cated in batch dim
            # data shape : batch_size * seq_len * data_shape
            cat_states = torch.cat(states, 0)
            cat_actions = torch.cat(actions, 0)
            cat_rewards = torch.cat(rewards, 0)
            critic_loss, actor_loss = self._traj_len_equal_learn(
                states=cat_states,
                actions=cat_actions,
                rewards=cat_rewards,
            )
        else:
            for i in range(batch_size):
                # because we learn element independetly
                # each element are 1 * seq_len * data_shape
                # so could use equal learn method
                critic_loss_temp, actor_loss_temp = self._traj_len_equal_learn(
                    states=states[i],
                    actions=actions[i],
                    rewards=rewards[i],
                )
                critic_loss += critic_loss_temp
                actor_loss += actor_loss_temp

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.update_count += 1
        if self.update_count > self.update_period:
            self.update_count = 0
            self.critic_target_net.load_state_dict(
                self.critic_net.state_dict())
        return critic_loss.item()

    def _traj_len_equal_learn(self,
                              states: torch.tensor,
                              actions: torch.tensor,
                              rewards: torch.tensor):
        # states; actions and rewards shape
        # are batch_size * seq_length * data_space
        states.to(self.device)
        actions.to(self.device)
        rewards.to(self.device)

        q_vals = self.critic_net(states)

        # shape : batch * seq * 1
        selected_q_v = q_vals.gather(
            2, actions).to(self.device)

        # shape : batch * seq * 1
        next_sel_q_v = torch.zeros_like(
            selected_q_v, device=self.device)
        next_sel_q_v[:-1] = self.critic_target_net(
            states[1:]).gather(
            2, actions[1:])
        target_values = (next_sel_q_v * self.discount_factor
                         + rewards)
        critic_loss = self._compute_critic_loss(
            selected_q_v, target_values.detach())

        advantage = 0.0
        action_prob = self.actor_net(states)
        if self.advantage_type == AdvantageTypes.QMinusV:
            state_values = torch.sum(
                q_vals * action_prob,
                dim=2).unsqueeze(-1)
            advantage = selected_q_v - state_values
        elif self.advantage_type == AdvantageTypes.RewardToGO:
            advantage = compute_reward_to_go(
                rewards, self.device) - selected_q_v
        else:
            raise ValueError(f"advantage type invalid {self.advantage_type}")
        log_prob = torch.log(action_prob).gather(2, actions)
        # because default alg is gradient desecent.
        # negative turn it into gradient ascendent
        actor_loss = (-torch.mean(log_prob *
                                  advantage.detach()))
        return critic_loss, actor_loss

    def _compute_critic_loss(self, input, target):
        return torch.mean((input - target) ** 2)

    def get_weight(self):
        weight = {
            "net": {
                "actor": self.actor_net.state_dict(),
                "critic": self.critic_net.state_dict(),
            },
            "optimizer": {
                "actor": self.actor_optim.state_dict(),
                "critic": self.critic_optim.state_dict(),
            }
        }
        return weight

    def set_weight(self, weight):
        net_w = weight["net"]
        optimizer_w = weight["optimizer"]
        self.actor_net.load_state_dict(net_w["actor"])
        self.critic_net.load_state_dict(net_w["critic"])
        self.actor_optim.load_state_dict(optimizer_w["actor"])
        self.critic_optim.load_state_dict(optimizer_w["critic"])

    def get_config(self):
        config = {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "update_period": self.update_period,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }
        return config

    def unwrapped(self):
        return self