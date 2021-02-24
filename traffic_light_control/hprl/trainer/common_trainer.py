from hprl.util.enum import TrainnerTypes
from hprl.util.checkpointer import Checkpointer
from typing import Any, Dict, List

import torch

from hprl.util.typing import Reward, TrainingRecord
from hprl.trainer.core import Train_Fn_Type, Trainer
from hprl.env import MultiAgentEnv
from hprl.policy import Policy
from hprl.replaybuffer import ReplayBuffer


class CommonTrainer(Trainer):
    def __init__(self,
                 type: TrainnerTypes,
                 config: Dict,
                 train_fn: Train_Fn_Type,
                 env: MultiAgentEnv,
                 policy: Policy,
                 replay_buffer: ReplayBuffer,
                 checkpointer: Checkpointer,
                 cumulative_train_iteration: int = 0) -> None:

        self.type = type
        self.config = config
        self.env = env
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.train_fn = train_fn
        self.checkpointer = checkpointer
        self.cumulative_train_iteration = max(0, cumulative_train_iteration)

    def train(self, episode: int) -> TrainingRecord:

        record = TrainingRecord({})

        for ep in range(episode):
            rewards = self.train_fn(
                self.env, self.policy,
                self.replay_buffer, self.config)
            self.cumulative_train_iteration += 1

            unwarp_r = self._unwrap_reward(rewards)
            print(unwarp_r)
            record.rewards[ep] = unwarp_r

            self.checkpointer.periodic_save(
                data=self.get_checkpoint(),
                iteration=self.cumulative_train_iteration,
            )

        return record

    def eval(self, episode: int) -> TrainingRecord:

        record = TrainingRecord({})

        for ep in episode:
            state = self.env.reset()
            rewards = []

            while True:
                action = self.policy.compute_action(state)
                ns, r, done, _ = self.env.step(action)
                state = ns[0]
                rewards.append(r[0])
                if done:
                    break

            record.rewards[ep] = self._unwrap_reward(rewards)
        return record

    def set_weight(self, weight: Dict):
        policy_w = weight.get("policy")
        buffer_w = weight.get("buffer")
        self.policy.set_weight(policy_w)
        self.replay_buffer.set_weight(buffer_w)

    def get_checkpoint(self):
        self.config["trained_iteration"] = self.cumulative_train_iteration
        config = {
            "type": self.type,
            "policy": self.policy.get_config(),
            "buffer": self.replay_buffer.get_config(),
            "executing": self.config,
        }
        weight = {
            "policy": self.policy.get_weight(),
            "buffer": self.replay_buffer.get_weight()
        }

        checkpoint = {
            "config": config,
            "weight": weight,
        }
        return checkpoint

    def save_checkpoint(self, checkpoint_dir: str = None, filename: str = None):
        checkpoint = self.get_checkpoint()
        checkpointer = self.checkpointer
        if checkpoint_dir:
            checkpointer = Checkpointer(checkpoint_dir)
        if not filename:
            checkpointer.periodic_save(
                self.cumulative_train_iteration, checkpoint)
        else:
            checkpointer.save(checkpoint, filename)

    def log_result(self, log_dir: str):
        raise NotImplementedError

    def _unwrap_reward(self, rewards: List[Reward]):
        length = len(rewards)
        ret = Reward(central=0.0, local={})
        if length == 0:
            return ret
        agents_id = self.env.get_agents_id()
        for k in agents_id:
            ret.local[k] = 0.0

        for r in rewards:
            ret.central += r.central
            for k, v in r.local.items():
                ret.local[k] += v

        ret.central /= length
        for k in r.local.keys():
            r.local[k] /= length
        return ret
