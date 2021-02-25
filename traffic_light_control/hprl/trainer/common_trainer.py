from hprl.util.plt import save_fig
import logging

from hprl.util.enum import TrainnerTypes
from hprl.util.checkpointer import Checkpointer
from typing import Any, Dict, List

from hprl.util.typing import Reward, TrainingRecord
from hprl.trainer.core import Log_Record_Fn_Type, Train_Fn_Type, Trainer
from hprl.env import MultiAgentEnv
from hprl.policy import Policy
from hprl.replaybuffer import ReplayBuffer
from hprl.trainer.support_fn import _cal_cumulative_reward, _cal_avg_reward


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
        self.train_records: Dict[int, TrainingRecord] = {}
        self.eval_records: Dict[int, TrainingRecord] = {}

    def train(self, episode: int) -> Dict[int, TrainingRecord]:

        for ep in range(episode):
            self.logger.info("=== train episode {} begin ===".format(ep))
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
            self.logger.info("=== train episode {} end   ===".format(ep))

        return self.train_records

    def eval(self, episode: int) -> Dict[int, TrainingRecord]:

        # like DQN policy is wrapped by eplison greedy
        # but eplison greedy should not be applied at eval time
        # the unwrapped function unwrap the policy
        # if policy is not wrapped, method should return itself
        eval_policy = self.policy.unwrapped()

        for ep in range(episode):
            self.logger.info("+++ eval episode {} begin +++".format(ep))
            state = self.env.reset()
            rewards = []
            infos = []

            while True:
                action = eval_policy.compute_action(state)
                ns, r, done, info = self.env.step(action)

                state = ns

                rewards.append(r)
                infos.append(info)
                if done.central:
                    break

            record = TrainingRecord(rewards, infos)
            self.eval_records[ep] = record
            self.log_record_fn(record, self.logger)

            self.logger.info("+++ eval episode {} end   +++".format(ep))

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
        culumative_rewards: List[Reward] = []
        for r in self.train_records.values():
            culumative_rewards.append(
                _cal_cumulative_reward(r.rewards)
            )
        central_reward = []

        agents_id = self.env.get_agents_id()
        local_reward = {id_: [] for id_ in agents_id}
        for r in culumative_rewards:
            central_reward.append(r.central)
            for id in agents_id:
                local_reward[id].append(r.local[id])
        episodes = list(self.train_records.keys())
        save_fig(
            x=episodes,
            y=central_reward,
            x_lable="episodes",
            y_label="training central reward",
            title="training central reward",
            dir=log_dir,
            img_name="training_central_reward",
        )
        for id in agents_id:
            save_fig(
                x=episodes,
                y=local_reward[id],
                x_lable="episodes",
                y_label="training local {} reward".format(id),
                title="training local {} reward".format(id),
                dir=log_dir,
                img_name="training_local_{}_reward".format(id),
            )
