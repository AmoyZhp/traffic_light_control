import abc
import logging
import os
import time
from typing import Callable, Dict, List

from hprl.env import MultiAgentEnv
from hprl.policy.policy import MultiAgentPolicy, PolicyTypes
from hprl.replaybuffer import MAgentReplayBuffer, ReplayBufferTypes
from hprl.trainer.recording import log_record, read_ckpt, write_ckpt
from hprl.trainer.trainer import Trainer
from hprl.util.typing import TrainingRecord, Transition

logger = logging.getLogger(__name__)


class BasisTrainer(Trainer):
    def __init__(
        self,
        type: PolicyTypes,
        env: MultiAgentEnv,
        policy: MultiAgentPolicy,
        config: Dict,
        trained_iteration=0,
        records: List[TrainingRecord] = [],
        output_dir: str = "",
    ):
        if output_dir and not os.path.exists(output_dir):
            raise ValueError(
                "output directory {} not exists".format(output_dir))

        self._type = type
        self._env = env
        self._policy = policy
        self._config = config

        self._records = records
        self._output_dir = output_dir
        self._trained_iteration = trained_iteration
        self._agents_id = self._env.agents_id

    @abc.abstractmethod
    def _train(self, ep: int, episodes: int):
        ...

    def train(
        self,
        episodes: int,
        ckpt_frequency: int = 0,
        log_record_fn: Callable = log_record,
    ):
        for ep in range(1, episodes + 1):
            self._trained_iteration += 1
            logger.info(f"========== train episode {ep} begin ==========", )
            logger.info("total trained iteration : %d",
                        self._trained_iteration)
            # implement _train in subclass
            record = self._train(ep, episodes)
            record.set_episode(self._trained_iteration)

            log_record_fn(record=record, logger=logger)
            self._records.append(record)
            if (ckpt_frequency > 0 and ep % ckpt_frequency == 0):
                self.save_checkpoint()
            logger.info("========= train end   =========")
        return self._records

    def eval(
        self,
        episodes: int,
        log_record_fn: Callable = log_record,
    ) -> List[TrainingRecord]:
        eval_policy = self._policy.unwrapped()
        records = []
        for ep in range(episodes):
            logger.info("========= eval episode {} begin =========".format(ep))

            rewards = []
            infos = []
            state = self._env.reset()

            while True:
                action = eval_policy.compute_action(state)
                next_s, reward, done, info = self._env.step(action)
                state = next_s

                rewards.append(reward)
                infos.append(info)
                if done.central:
                    break

            record = TrainingRecord(episode=ep, rewards=rewards, infos=infos)
            log_record_fn(record=record, logger=logger)
            records.append(record)
            logger.info("========= eval episode {} end   =========".format(ep))
        return records

    def save_checkpoint(self, path=""):
        ckpt = self.get_checkpoint()
        if not path:
            path = f"ckpt_{self._trained_iteration}.pth"
            if self._output_dir:
                path = f"{self._output_dir}/{path}"
        write_ckpt(ckpt=ckpt, path=path)

    def load_checkpoint(self, path: str):
        ckpt = read_ckpt(path)
        weight = ckpt["weight"]
        records = ckpt["records"]
        policy_w = weight.get("policy")
        self._policy.set_weight(policy_w)
        self._records = records

    def get_checkpoint(self) -> Dict:
        policy_weight = self._policy.get_weight()
        weight = {
            "policy": policy_weight,
        }
        config = self.get_config()
        records = self._records
        checkpoint = {
            "config": config,
            "weight": weight,
            "records": records,
        }
        return checkpoint

    def set_weight(self, weight: Dict):
        self._policy.set_weight(weight["policy"])

    def get_weight(self):
        policy_weight = self._policy.get_weight()
        weight = {
            "policy": policy_weight,
        }
        return weight

    def get_config(self):
        policy_config = self._policy.get_config()
        trainer_config = self._config
        trainer_config.update({
            "type": self._type,
            "trained_iteration": self._trained_iteration,
            "output_dir": self._output_dir
        })
        config = {
            "policy": policy_config,
            "trainer": trainer_config,
            "env": self._env.setting,
        }
        return config

    def set_records(self, records: List[TrainingRecord]):
        self._records = records

    def get_records(self):
        return self._records

    @property
    def output_dir(self):
        return self._output_dir


class OffPolicyTrainer(BasisTrainer):
    def __init__(
        self,
        type: PolicyTypes,
        env: MultiAgentEnv,
        policy: MultiAgentPolicy,
        buffer: MAgentReplayBuffer,
        config: Dict,
        trained_iteration=0,
        records: List[TrainingRecord] = [],
        output_dir: str = "",
    ):
        super().__init__(
            type,
            env,
            policy,
            config,
            trained_iteration=trained_iteration,
            records=records,
            output_dir=output_dir,
        )
        self._buffer = buffer

    def _train(self, ep: int, episodes: int):
        batch_size = self._config["batch_size"]
        beta = 0
        buffer_type = self._buffer.type
        if buffer_type == ReplayBufferTypes.Prioritized:
            init_beta = self._config["per_beta"]
            beta = (ep + 1) / episodes * (1 - init_beta) + init_beta

        sim_cost = 0.0
        learn_cost = 0.0
        state = self._env.reset()
        record = TrainingRecord()
        while True:
            action = self._policy.compute_action(state)

            sim_begin = time.time()
            next_s, r, done, info = self._env.step(action)
            sim_cost += time.time() - sim_begin
            transition = Transition(
                state=state,
                action=action,
                reward=r,
                next_state=next_s,
                terminal=done,
            )
            self._buffer.store(transition)
            state = next_s

            learn_begin = time.time()
            sample_data = self._buffer.sample(batch_size, beta)
            learn_info = self._policy.learn_on_batch(sample_data)
            if buffer_type == ReplayBufferTypes.Prioritized:
                priorities = learn_info.get("priorities", [])
                self._buffer.update_priorities(
                    idxes=sample_data.idxes,
                    priorities=priorities,
                )
            learn_cost += time.time() - learn_begin

            record.append_reward(r)
            record.append_info(info)

            if done.central:
                logger.info("simulation time cost : {:.3f}s".format(sim_cost))
                logger.info("learning time cost : {:.3f}s".format(learn_cost))
                break
        return record

    def get_checkpoint(self):
        checkpoint = super().get_checkpoint()
        buffer_weight = self._buffer.get_weight()
        checkpoint["weight"]["buffer"] = buffer_weight
        return checkpoint

    def load_checkpoint(self, path: str):
        ckpt = read_ckpt(path)
        weight = ckpt["weight"]
        records = ckpt["records"]
        self._policy.set_weight(weight["policy"])
        self._buffer.set_weight(weight["buffer"])
        self._records = records

    def get_config(self):
        config = super().get_config()
        buffer_config = self._buffer.get_config()
        config["buffer"] = buffer_config
        return config

    def set_weight(self, weight: Dict):
        self._buffer.set_weight(weight["buffer"])
        super().set_weight(weight)

    def get_weight(self):
        weight = super().get_weight()
        buffer_weight = self._buffer.get_weight()
        weight["buffer"] = buffer_weight
        return weight
