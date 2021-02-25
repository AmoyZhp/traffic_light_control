
from hprl.util.checkpointer import Checkpointer
from typing import Dict
from hprl.util.enum import TrainnerTypes
from hprl.util.typing import TrainingRecord
import torch.nn as nn

from hprl.trainer.core import Log_Record_Fn_Type, Train_Fn_Type
from hprl.trainer.common_trainer import CommonTrainer
from hprl.env import MultiAgentEnv
from hprl.policy import Policy
from hprl.replaybuffer import ReplayBuffer


class QLearningTranier(CommonTrainer):
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
        super().__init__(
            type=type,
            config=config,
            train_fn=train_fn,
            env=env,
            policy=policy,
            replay_buffer=replay_buffer,
            checkpointer=checkpointer,
            log_record_fn=log_record_fn,
            cumulative_train_iteration=cumulative_train_iteration
        )

    def eval(self, episode: int):
        record = TrainingRecord({})

        for ep in episode:
            state = self.env.reset()
            rewards = []

            while True:
                raw_q_policy = self.policy.unwrapped()
                action = raw_q_policy.compute_action(state)
                ns, r, done, _ = self.env.step(action)
                state = ns[0]
                rewards.append(r[0])
                if done:
                    break

            record.rewards[ep] = self._unwrap_reward(rewards)
        return record
