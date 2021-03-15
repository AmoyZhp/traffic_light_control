from hprl.trainer.independent_learner_trainer import IndependentLearnerTrainer, off_policy_train_fn
from hprl.recorder.torch_recorder import TorchRecorder
import logging
from hprl.env.multi_agent_env import MultiAgentEnv
from hprl.policy.decorator.epsilon_greedy import SingleEpsilonGreedy
from hprl.replaybuffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from hprl.util.enum import ReplayBufferTypes, TrainnerTypes
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from hprl.util.typing import ExecutingConfig, SampleBatch, Transition, TransitionTuple
from hprl.policy.policy import SingleAction, SingleAgentPolicy

logger = logging.getLogger(__name__)


def build_iql_trainer(config, env: MultiAgentEnv, models):
    buffer_config = config["buffer"]
    policy_config = config["policy"]
    executing_config = config["executing"]

    agents_id = env.get_agents_id()
    policies = {}
    for id_ in agents_id:
        model = models[id_]
        inner_p = PERDQN(
            acting_net=model["acting_net"],
            target_net=model["target_net"],
            critic_lr=policy_config["critic_lr"],
            discount_factor=policy_config["discount_factor"],
            update_period=policy_config["update_period"],
            action_space=policy_config["action_space"][id_],
            state_space=policy_config["state_space"][id_],
        )
        policies[id_] = SingleEpsilonGreedy(
            inner_policy=inner_p,
            eps_frame=policy_config["eps_frame"],
            eps_min=policy_config["eps_min"],
            eps_init=policy_config["eps_init"],
            action_space=policy_config["action_space"][id_],
        )
    buffer_type = buffer_config["type"]
    buffers = {}
    if buffer_type == ReplayBufferTypes.Prioritized:
        capacity = buffer_config["capacity"]
        alpha = buffer_config["alpha"]
        for id in agents_id:
            buffers[id] = PrioritizedReplayBuffer(
                capacity=capacity,
                alpha=alpha,
            )
    recorder = TorchRecorder(executing_config["record_base_dir"])
    trainer = IndependentLearnerTrainer(
        type=TrainnerTypes.IQL_PER,
        policies=policies,
        replay_buffers=buffers,
        env=env,
        train_fn=off_policy_train_fn,
        recorder=recorder,
        config=executing_config,
    )
    return trainer


class PERDQN(SingleAgentPolicy):
    def __init__(
        self,
        acting_net: nn.Module,
        target_net: nn.Module,
        critic_lr: int,
        discount_factor: float,
        update_period: int,
        action_space,
        state_space,
        device=None,
    ) -> None:

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.acting_net = acting_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.acting_net.state_dict())

        self.target_net.to(self.device)
        self.acting_net.to(self.device)

        self.optimizer = optim.Adam(self.acting_net.parameters(), critic_lr)
        self.loss_func = self._loss_func
        self.update_count = 0
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.update_period = update_period
        self.action_space = action_space
        self.state_space = state_space

    def compute_action(self, state: np.ndarray) -> SingleAction:
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            value = self.acting_net(state)
            value = np.squeeze(value, 0)
        _, index = torch.max(value, 0)
        action = index.item()
        return action

    def learn_on_batch(self, sample_batch: SampleBatch) -> List[float]:
        if sample_batch is None:
            return []

        batch_trans = sample_batch.transitions
        if not batch_trans:
            return []
        batch_size = len(batch_trans)
        # from List[TransitionTuple] to be
        # TransitionTuple(List[state], List[action], ..., dypte=torch.tensor)
        batch_trans = self._np_to_torch(batch_trans)

        mask_batch = torch.tensor(
            tuple(map(lambda d: not d, batch_trans.terminal)),
            device=self.device,
            dtype=torch.bool,
        )
        s_batch = torch.cat(batch_trans.state, 0).to(self.device)
        a_batch = torch.cat(batch_trans.action, 0).to(self.device)
        r_batch = torch.cat(batch_trans.reward, 0).to(self.device)
        next_s_batch = torch.cat(batch_trans.next_state, 0).to(self.device)
        logger.debug(f"s shape {s_batch.shape}")
        logger.debug(f"a shape {a_batch.shape}")
        logger.debug(f"r shape {r_batch.shape}")
        logger.debug(f"mask batch shape {mask_batch.shape}")

        q_vals = self.acting_net(s_batch).gather(1, a_batch)
        q_vals = q_vals.to(self.device)

        next_q_vals = torch.zeros(batch_size, device=self.device)
        next_q_vals[mask_batch] = self.target_net(
            next_s_batch[mask_batch]).max(1)[0]
        next_q_vals = next_q_vals.unsqueeze(1)

        expected_q_vals = (next_q_vals * self.discount_factor) + r_batch
        td_error = expected_q_vals - q_vals
        td_error = td_error.squeeze(-1)
        logger.debug(f"td error shape {td_error.shape}")
        weights = torch.tensor(sample_batch.weights,
                               dtype=torch.float).unsqueeze(-1)
        logger.debug(f"expected q vals shape {expected_q_vals.shape}")
        logger.debug(f"weigth shape {weights.shape}")
        loss = self.loss_func(
            q_vals,
            expected_q_vals.detach(),
            weights,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.update_period == 0:
            self.target_net.load_state_dict(self.acting_net.state_dict())
            self.update_count = 0

        priorities = torch.clamp(td_error, -1, 1).detach()
        priorities += abs(priorities) + 1e-8
        priorities = priorities.tolist()
        return priorities

    def _loss_func(self, input, target, weight):
        computed = ((input - target)**2) * weight
        return torch.mean(computed)

    def _np_to_torch(self, batch_data: List[TransitionTuple]):
        def np_to_torch(trans: TransitionTuple):
            torch_trans = TransitionTuple(
                torch.tensor(trans.state, dtype=torch.float).unsqueeze(0),
                torch.tensor(trans.action, dtype=torch.long).view(-1, 1),
                torch.tensor(trans.reward, dtype=torch.float).view(-1, 1),
                torch.tensor(trans.next_state, dtype=torch.float).unsqueeze(0),
                torch.tensor(trans.terminal, dtype=torch.long))
            return torch_trans

        batch_data = map(np_to_torch, batch_data)
        batch = TransitionTuple(*zip(*batch_data))
        return batch

    def get_weight(self):
        weight = {
            "net": self.acting_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return weight

    def set_weight(self, weight):
        net_w = weight["net"]
        optimizer_w = weight["optimizer"]
        self.acting_net.load_state_dict(net_w)
        self.target_net.load_state_dict(net_w)
        self.optimizer.load_state_dict(optimizer_w)
        self.step = weight["step"]

    def get_config(self):
        config = {
            "critic_lr": self.critic_lr,
            "discount_factor": self.discount_factor,
            "update_period": self.update_period,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }
        return config

    def unwrapped(self):
        return self
