import abc
import time
from hprl.replaybuffer.replay_buffer import ReplayBufferTypes
from hprl.util.typing import TrainingRecord, Transition
import logging
from typing import Dict

from hprl.trainer.trainer import Trainer
from hprl.policy import PolicyTypes, MultiAgentPolicy
from hprl.replaybuffer import MultiAgentReplayBuffer
from hprl.env import MultiAgentEnv
from hprl.recorder import Recorder

logger = logging.getLogger(__package__)


class MultiAgentTrainer(Trainer):
    """
        MultiAgentTrainer can't be used directly
        inherit it and implement 
            _train(ep:int, episodes:int) -> TrainingRecord 
        func in subclass

    Args:
        Trainer ([type]): [description]
    """
    def __init__(
        self,
        type: PolicyTypes,
        config: Dict,
        env: MultiAgentEnv,
        policy: MultiAgentPolicy,
        recorder: Recorder,
    ) -> None:
        self.type = type
        self.config = config
        self.env = env
        self.policy = policy
        self.recorder = recorder
        self.agents_id = self.env.get_agents_id()
        self.trained_iteration = 0

    def train(self, episodes: int):
        ckpt_frequency = self.config.get("ckpt_frequency", 0)
        if ckpt_frequency <= 0:
            logger.info(
                "checkpoint saved frequnecy is zero, "
                "therefore checkpoint file will not be saved during training")
        for ep in range(episodes):
            logger.info("========== train episode {} begin ==========".format(
                self.trained_iteration))
            # implement _train in subclass
            record = self._train(ep, episodes)
            self.trained_iteration += 1
            record.set_episode(self.trained_iteration)
            fig = True if (ep + 1) % (episodes / 10) == 0 else False
            self.recorder.print_record(
                record=record,
                logger=logger,
                fig=fig,
            )
            if (ckpt_frequency > 0
                    and self.trained_iteration % ckpt_frequency == 0):
                self.recorder.write_ckpt(
                    ckpt=self.get_checkpoint(),
                    filename=f"ckpt_{self.trained_iteration}.pth",
                )
                self.recorder.write_records()
            logger.info("========= train episode {} end   =========".format(
                self.trained_iteration))

    @abc.abstractmethod
    def _train(self, ep: int, episodes: int):
        ...

    def eval(self, episodes: int):
        eval_policy = self.policy.unwrapped()

        for ep in range(episodes):
            logger.info("+++++++++ eval episode {} begin +++++++++".format(ep))

            rewards = []
            infos = []

            state = self.env.reset()
            while True:
                action = eval_policy.compute_action(state)
                next_s, r, done, info = self.env.step(action)
                state = next_s

                rewards.append(r)
                infos.append(info)
                if done.central:
                    break
            record = TrainingRecord(ep, rewards, infos)
            self.recorder.print_record(record, logger, False)
            logger.info("+++++++++ eval episode {} end   +++++++++".format(ep))

    def get_checkpoint(self):
        policy_weight = self.policy.get_weight()
        weight = {
            "policy": policy_weight,
        }
        config = self.get_config()
        records = self.recorder.get_records()
        checkpoint = {
            "config": config,
            "weight": weight,
            "records": records,
        }
        return checkpoint

    def load_checkpoint(self, checkpoint: Dict):
        weight = checkpoint["weight"]
        records = checkpoint["records"]
        policy_w = weight.get("policy")
        self.policy.set_weight(policy_w)
        self.recorder.add_records(records)

    def get_config(self):
        policy_config = self.policy.get_config()
        config = {
            "type": self.type,
            "trained_iteration": self.trained_iteration,
            "policy": policy_config,
            "executing": self.config,
        }
        return config

    def get_records(self):
        self.recorder.get_records()

    def save_config(self, dir: str = ""):
        self.recorder.write_config(self.get_config(), dir)

    def save_records(self, dir: str = ""):
        self.recorder.write_records(dir)

    def save_checkpoint(self, dir: str = "", filename: str = ""):
        self.recorder.write_ckpt(self.get_checkpoint(), dir, filename)


class OffPolicy(MultiAgentTrainer):
    def __init__(
        self,
        type: PolicyTypes,
        config: Dict,
        env: MultiAgentEnv,
        policy: MultiAgentPolicy,
        buffer: MultiAgentReplayBuffer,
        recorder: Recorder,
    ) -> None:
        super().__init__(
            type,
            config,
            env,
            policy,
            recorder,
        )
        self.buffer = buffer

    def _train(self, ep: int, episodes: int):
        batch_size = self.config["batch_size"]
        beta = 0
        buffer_type = self.buffer.type
        if buffer_type == ReplayBufferTypes.Prioritized:
            init_beta = self.config["per_beta"]
            beta = (ep + 1) / episodes * (1 - init_beta) + init_beta

        sim_cost = 0.0
        learn_cost = 0.0
        state = self.env.reset()
        record = TrainingRecord()
        while True:
            action = self.policy.compute_action(state)

            sim_begin = time.time()
            next_s, r, done, info = self.env.step(action)
            sim_cost += time.time() - sim_begin
            transition = Transition(
                state=state,
                action=action,
                reward=r,
                next_state=next_s,
                terminal=done,
            )
            self.buffer.store(transition)
            state = next_s

            learn_begin = time.time()
            sample_data = self.buffer.sample(batch_size, beta)
            info = self.policy.learn_on_batch(sample_data)
            if buffer_type == ReplayBufferTypes.Prioritized:
                priorities = info.get("priorities", [])
                self.buffer.update_priorities(
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
        buffer_weight = self.buffer.get_weight()
        checkpoint["weight"]["buffer"] = buffer_weight
        return checkpoint

    def load_checkpoint(self, checkpoint: Dict):
        weight = checkpoint["weight"]
        buffer_w = weight.get("buffer")
        self.buffer.set_weight(buffer_w)
        super().load_checkpoint(checkpoint)

    def get_config(self):
        config = super().get_config()
        buffer_config = self.buffer.get_config()
        config["buffer"] = buffer_config
        return config