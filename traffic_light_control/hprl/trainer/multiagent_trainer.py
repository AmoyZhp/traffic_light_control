from hprl.util.typing import TrainingRecord, Trajectory, Transition
import logging
import time
from typing import Dict
from hprl.trainer.trainer import Trainer
from hprl.env import MultiAgentEnv
from hprl.policy import MultiAgentPolicy
from hprl.replaybuffer import MultiAgentReplayBuffer
from hprl.recorder import Recorder
from hprl.policy.policy import PolicyTypes

logger = logging.getLogger(__package__)


def on_policy_train_fn(
    config: Dict,
    env: MultiAgentEnv,
    policy: MultiAgentPolicy,
    buffer: MultiAgentReplayBuffer,
    logger: logging.Logger,
):
    batch_size = config["batch_size"]
    sim_cost = 0.0
    learn_cost = 0.0
    sample_data = []
    for _ in range(batch_size):
        state = env.reset()
        record = TrainingRecord()
        states = []
        rewards = []
        actions = []
        while True:
            states.append(state)
            action = policy.compute_action(state)

            sim_begin = time.time()
            next_s, r, done, info = env.step(action)
            sim_cost += time.time() - sim_begin
            state = next_s

            actions.append(action)
            rewards.append(r)

            record.append_reward(r)
            record.append_info(info)

            if done.central:
                traj = Trajectory(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    terminal=done.central,
                )
                sample_data.append(traj)
                break
        learn_begin = time.time()
        sample_data = buffer.sample(batch_size)
        policy.learn_on_batch(sample_data)
        learn_cost += time.time() - learn_begin
        logger.info("simulation time cost : {:.3f}s".format(sim_cost))
        logger.info("learning time cost : {:.3f}s".format(learn_cost))
    return record


def off_policy_train_fn(
    config: Dict,
    env: MultiAgentEnv,
    policy: MultiAgentPolicy,
    buffer: MultiAgentReplayBuffer,
    logger: logging.Logger,
):
    batch_size = config["batch_size"]
    sim_cost = 0.0
    learn_cost = 0.0

    state = env.reset()
    record = TrainingRecord()
    while True:
        action = policy.compute_action(state)

        sim_begin = time.time()
        next_s, r, done, info = env.step(action)
        sim_cost += time.time() - sim_begin
        transition = Transition(
            state=state,
            action=action,
            reward=r,
            next_state=next_s,
            terminal=done,
        )
        buffer.store(transition)
        state = next_s

        learn_begin = time.time()
        sample_data = buffer.sample(batch_size)
        policy.learn_on_batch(sample_data)
        learn_cost += time.time() - learn_begin

        record.append_reward(r)
        record.append_info(info)

        if done.central:
            logger.info("simulation time cost : {:.3f}s".format(sim_cost))
            logger.info("learning time cost : {:.3f}s".format(learn_cost))
            break
    return record


def off_policy_per_train_fn(
    config: Dict,
    env: MultiAgentEnv,
    policy: MultiAgentPolicy,
    buffer: MultiAgentReplayBuffer,
    logger: logging.Logger,
):
    batch_size = config["batch_size"]
    beta = config["per_beta"]
    sim_cost = 0.0
    learn_cost = 0.0

    state = env.reset()
    record = TrainingRecord()
    while True:
        action = policy.compute_action(state)

        sim_begin = time.time()
        next_s, r, done, info = env.step(action)
        sim_cost += time.time() - sim_begin
        transition = Transition(
            state=state,
            action=action,
            reward=r,
            next_state=next_s,
            terminal=done,
        )
        buffer.store(transition)
        state = next_s

        learn_begin = time.time()
        sample_data = buffer.sample(batch_size, beta)
        info = policy.learn_on_batch(sample_data)
        priorities = info.get("priorities", [])
        buffer.update_priorities(
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


class MultiAgentTraienr(Trainer):
    def __init__(
        self,
        type: PolicyTypes,
        config: Dict,
        env: MultiAgentEnv,
        policy: MultiAgentPolicy,
        replay_buffer: MultiAgentReplayBuffer,
        train_fn,
        recorder: Recorder,
    ) -> None:
        logger.info("%s trainer start init", type.value)
        self.type = type
        self.config = config
        self.env = env
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.train_fn = train_fn
        self.recorder = recorder
        self.agents_id = self.env.get_agents_id()
        self.trained_iteration = 0
        logger.info("ckpt frequency : %d", self.config["ckpt_frequency"])
        logger.info("per beta : %f", self.config["per_beta"])
        logger.info("train fn name : %s", self.train_fn.__name__)
        logger.info("%s trainer start init done", type.value)

    def train(self, episodes: int):
        ckpt_frequency = self.config["ckpt_frequency"]
        init_beta = self.config["per_beta"]
        left_beta = 1.0 - init_beta
        for ep in range(episodes):
            logger.info("========== train episode {} begin ==========".format(
                self.trained_iteration))
            # beta increased by trained iteration
            # it will equal one in the end
            self.config["per_beta"] = (ep +
                                       1) / episodes * left_beta + init_beta
            record = self.train_fn(
                self.config,
                self.env,
                self.policy,
                self.replay_buffer,
                logger,
            )
            self.trained_iteration += 1
            record.set_episode(self.trained_iteration)
            self.recorder.add_record(record)
            fig = True if (ep + 1) % (episodes / 10) == 0 else False
            self.recorder.print_record(
                record=record,
                logger=logger,
                fig=fig,
            )
            if (ckpt_frequency != 0
                    and self.trained_iteration % ckpt_frequency == 0):
                self.recorder.write_ckpt(
                    ckpt=self.get_checkpoint(),
                    filename=f"ckpt_{self.trained_iteration}.pth",
                )
                self.recorder.write_records()
            logger.info("========= train episode {} end   =========".format(
                self.trained_iteration))

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
        buffer_weight = self.replay_buffer.get_weight()
        weight = {
            "policy": policy_weight,
            "buffer": buffer_weight,
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
        buffer_w = weight.get("buffer")
        self.policy.set_weight(policy_w)
        self.replay_buffer.set_weight(buffer_w)
        self.recorder.add_records(records)

    def get_config(self):
        policy_config = self.policy.get_config()
        buffer_config = self.replay_buffer.get_config()
        config = {
            "type": self.type,
            "trained_iteration": self.trained_iteration,
            "policy": policy_config,
            "buffer": buffer_config,
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