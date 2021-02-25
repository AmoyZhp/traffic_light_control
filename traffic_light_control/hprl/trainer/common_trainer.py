import logging

from numpy.lib.utils import info
from trainer.independent_traffic_trainer import RECORDS_ROOT_DIR
from hprl.util.enum import TrainnerTypes
from hprl.util.checkpointer import Checkpointer
from typing import Any, Dict, List

import torch

from hprl.util.typing import Reward, TrainingRecord
from hprl.trainer.core import Log_Record_Fn_Type, Train_Fn_Type, Trainer
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
                 log_record_fn: Log_Record_Fn_Type,
                 cumulative_train_iteration: int = 0) -> None:

        self.type = type
        self.config = config
        self.env = env
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.train_fn = train_fn
        self.checkpointer = checkpointer
        self.log_record_fn = log_record_fn
        self.cumulative_train_iteration = max(0, cumulative_train_iteration)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.train_records = {}
        self.eval_records = {}

    def train(self, episode: int) -> Dict[int, TrainingRecord]:

        for ep in range(episode):
            self.logger.info("=== episode {} begin ===".format(ep))
            rewards, infos = self.train_fn(
                self.env,
                self.policy,
                self.replay_buffer,
                self.config,
                self.logger)
            self.cumulative_train_iteration += 1

            record = TrainingRecord(rewards, infos)
            self.train_records[ep] = record

            self.log_record_fn(record, self.logger)

            self.checkpointer.periodic_save(
                data=self.get_checkpoint(),
                iteration=self.cumulative_train_iteration,
            )
            self.logger.info("=== episode {} end   ===".format(ep))

        return self.train_records

    def eval(self, episode: int) -> Dict[int, TrainingRecord]:

        for ep in episode:
            state = self.env.reset()
            rewards = []
            infos = []

            while True:
                action = self.policy.compute_action(state)
                ns, r, done, info = self.env.step(action)

                state = ns

                rewards.append(r)
                infos.append(info)

                if done:
                    break

            record = TrainingRecord(rewards, infos)
            self.eval_records[ep] = record
            self.log_record_fn(record, self.logger)

        return self.eval_records

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

    def _unwrap_reward(self, rewards: List[Reward]) -> Reward:
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