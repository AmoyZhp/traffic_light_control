from enum import Enum
import json
import logging

from hprl.util.enum import TrainnerTypes
from hprl.util.checkpointer import Checkpointer
from typing import Dict, List

from hprl.util.typing import Reward, TrainingRecord
from hprl.trainer.trainer import Log_Record_Fn_Type, Train_Fn_Type, Trainer
from hprl.env import MultiAgentEnv
from hprl.policy import Policy
from hprl.replaybuffer import ReplayBuffer
from hprl.util.support_fn import cal_cumulative_reward, cal_avg_reward
from hprl.util.plt import save_fig


class CommonTrainer(Trainer):
    def __init__(
        self,
        type: TrainnerTypes,
        config: Dict,
        train_fn: Train_Fn_Type,
        env: MultiAgentEnv,
        policy: Policy,
        replay_buffer: ReplayBuffer,
        checkpointer: Checkpointer,
        log_record_fn: Log_Record_Fn_Type,
        record_base_dir: str,
        log_dir: str = None,
        cumulative_train_iteration: int = 0,
    ) -> None:

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
        self.log_dir = log_dir
        if log_dir is not None:
            file_handler = logging.FileHandler(f"{log_dir}/{__name__}.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.record_base_dir = record_base_dir

        self.train_records: Dict[int, TrainingRecord] = {}
        self.eval_records: Dict[int, Dict[int, TrainingRecord]] = {}

    def train(self, episode: int) -> Dict[int, TrainingRecord]:

        for ep in range(episode):
            self.logger.info(
                "========== train episode {} begin ==========".format(
                    self.cumulative_train_iteration))
            rewards, infos = self.train_fn(self.env, self.policy,
                                           self.replay_buffer, self.config,
                                           self.logger)
            self.cumulative_train_iteration += 1

            record = TrainingRecord(rewards, infos)
            self.train_records[self.cumulative_train_iteration] = record

            self.log_record_fn(record, self.logger)

            self.checkpointer.periodic_save(
                data=self.get_checkpoint(),
                iteration=self.cumulative_train_iteration,
            )
            self.log_records(self.log_dir)
            self.logger.info(
                "========= train episode {} end   =========".format(
                    self.cumulative_train_iteration))

        return self.train_records

    def eval(self, episode: int) -> Dict[int, TrainingRecord]:

        # like DQN policy is wrapped by eplison greedy
        # but eplison greedy should not be applied at eval time
        # the unwrapped function unwrap the policy
        # if policy is not wrapped, method should return itself
        eval_policy = self.policy.unwrapped()

        eval_records = {}
        for ep in range(episode):
            self.logger.info(
                "+++++++++ eval episode {} begin +++++++++".format(ep))
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
            eval_records[ep] = record
            self.log_record_fn(record, self.logger)

            self.logger.info(
                "+++++++++ eval episode {} end   +++++++++".format(ep))

        self.eval_records[self.cumulative_train_iteration] = eval_records

        return self.eval_records

    def load_checkpoint(self, checkpoint: Dict):
        weight = checkpoint["weight"]
        records = checkpoint["records"]

        policy_w = weight.get("policy")
        buffer_w = weight.get("buffer")
        self.policy.set_weight(policy_w)
        self.replay_buffer.set_weight(buffer_w)

        self.train_records = records["train_records"]
        self.eval_records = records["eval_records"]

    def get_config(self):
        config = {
            "type": self.type,
            "policy": self.policy.get_config(),
            "buffer": self.replay_buffer.get_config(),
            "executing": self.config,
        }
        return config

    def get_checkpoint(self):
        self.config["trained_iteration"] = self.cumulative_train_iteration
        config = self.get_config()
        weight = {
            "policy": self.policy.get_weight(),
            "buffer": self.replay_buffer.get_weight()
        }

        records = {
            "train_records": self.train_records,
            "eval_records": self.eval_records,
        }

        checkpoint = {
            "config": config,
            "weight": weight,
            "records": records,
        }
        return checkpoint

    def save_checkpoint(self,
                        checkpoint_dir: str = None,
                        filename: str = None):
        checkpoint = self.get_checkpoint()
        checkpointer = self.checkpointer
        if checkpoint_dir:
            checkpointer = Checkpointer(checkpoint_dir)
        if not filename:
            checkpointer.periodic_save(self.cumulative_train_iteration,
                                       checkpoint)
        else:
            checkpointer.save(checkpoint, filename)

    def log_records(self, log_dir: str):
        self._log_train_culumative_reward(log_dir)
        self._log_train_avg_reward(log_dir)
        self._log_eval_avg_reward(log_dir)
        self._log_eval_culumative_reward(log_dir)
        self._log_record(log_dir)

    def get_records(self):
        return {
            "train": self.train_records,
            "eval": self.eval_records,
        }

    def log_config(self, log_dir: str):
        config = self.get_config()
        config_path = f"{log_dir}/init_config.json"

        class EnumEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Enum):
                    return obj.name
                return json.JSONEncoder.default(self, obj)

        with open(config_path, "w") as f:
            json.dump(config, f, cls=EnumEncoder)

    def _log_record(self, log_dir: str):
        record_file = f"{log_dir}/records.txt"
        with open(record_file, "w", encoding="utf-8") as f:
            f.write(str(self.get_records()))

    def _log_train_culumative_reward(self, log_dir: str):
        culumative_rewards: List[Reward] = []
        for r in self.train_records.values():
            culumative_rewards.append(cal_cumulative_reward(r.rewards))
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
            y_label="reward",
            title="training culumative central reward",
            dir=log_dir,
            img_name="training_culumative_central_reward",
        )
        for id in agents_id:
            save_fig(
                x=episodes,
                y=local_reward[id],
                x_lable="episodes",
                y_label="reward",
                title="training culumative local {} reward".format(id),
                dir=log_dir,
                img_name="training_culumative_local_{}_reward".format(id),
            )

    def _log_train_avg_reward(self, log_dir: str):
        avg_reward: List[Reward] = []
        for r in self.train_records.values():
            avg_reward.append(cal_avg_reward(r.rewards))
        central_reward = []

        agents_id = self.env.get_agents_id()
        local_reward = {id_: [] for id_ in agents_id}
        for r in avg_reward:
            central_reward.append(r.central)
            for id in agents_id:
                local_reward[id].append(r.local[id])
        episodes = list(self.train_records.keys())
        save_fig(
            x=episodes,
            y=central_reward,
            x_lable="episodes",
            y_label="reward",
            title="training avg central reward",
            dir=log_dir,
            img_name="training_avg_central_reward",
        )
        for id in agents_id:
            save_fig(
                x=episodes,
                y=local_reward[id],
                x_lable="episodes",
                y_label="reward",
                title="training avg local {} reward".format(id),
                dir=log_dir,
                img_name="training_avg_local_{}_reward".format(id),
            )

    def _log_eval_avg_reward(self, log_dir: str):
        avg_reward: List[Reward] = []
        for one_eval_turn in self.eval_records.values():
            avg_reward_one_turn = []
            for r in one_eval_turn.values():
                avg_reward_one_turn.append(cal_avg_reward(r.rewards))
            avg_reward.append(cal_avg_reward(avg_reward_one_turn))
        central_reward = []

        agents_id = self.env.get_agents_id()
        local_reward = {id_: [] for id_ in agents_id}
        for r in avg_reward:
            central_reward.append(r.central)
            for id in agents_id:
                local_reward[id].append(r.local[id])
        episodes = list(self.eval_records.keys())
        save_fig(
            x=episodes,
            y=central_reward,
            x_lable="episodes",
            y_label="reward",
            title="eval avg central reward",
            dir=log_dir,
            img_name="eval_avg_central_reward",
        )
        for id in agents_id:
            save_fig(
                x=episodes,
                y=local_reward[id],
                x_lable="episodes",
                y_label="reward",
                title="eval avg local {} reward".format(id),
                dir=log_dir,
                img_name="eval_avg_local_{}_reward".format(id),
            )

    def _log_eval_culumative_reward(self, log_dir: str):
        culumative_reward: List[Reward] = []
        for one_eval_turn in self.eval_records.values():
            culumative_reward_one_turn = []
            for r in one_eval_turn.values():
                culumative_reward_one_turn.append(
                    cal_cumulative_reward(r.rewards))
            culumative_reward.append(
                cal_cumulative_reward(culumative_reward_one_turn))
        central_reward = []

        agents_id = self.env.get_agents_id()
        local_reward = {id_: [] for id_ in agents_id}
        for r in culumative_reward:
            central_reward.append(r.central)
            for id in agents_id:
                local_reward[id].append(r.local[id])
        episodes = list(self.eval_records.keys())
        save_fig(
            x=episodes,
            y=central_reward,
            x_lable="episodes",
            y_label="reward",
            title="eval culumative central reward",
            dir=log_dir,
            img_name="eval_culumative_central_reward",
        )
        for id in agents_id:
            save_fig(
                x=episodes,
                y=local_reward[id],
                x_lable="episodes",
                y_label="reward",
                title="eval culumative local {} reward".format(id),
                dir=log_dir,
                img_name="eval_culumative_local_{}_reward".format(id),
            )
