from hprl.replaybuffer.replay_buffer import ReplayBuffer
from hprl.recorder.recorder import Recorder
from hprl.util.enum import TrainnerTypes
import logging
import time
from hprl.util.typing import Action, ExecutingConfig, Reward, SampleBatch, State, Terminal, TrainingRecord, TrajectoryTuple, TransitionTuple
from hprl.env.multi_agent_env import MultiAgentEnv
from hprl.policy.policy import Policy
from hprl.replaybuffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from typing import Dict, List
from hprl.trainer.trainer import Trainer

logger = logging.getLogger(__package__)


def on_policy_train_fn(
    env: MultiAgentEnv,
    policies: Dict[str, Policy],
    buffers: Dict[str, ReplayBuffer],
    config: ExecutingConfig,
    logger: logging.Logger,
):
    batch_size = config["batch_size"]
    beta = config["per_beta"]
    sim_cost = 0.0
    learn_cost = 0.0

    agents_id = env.get_agents_id()
    samples_batch = {id: SampleBatch() for id in agents_id}
    for _ in range(batch_size):
        record = TrainingRecord()
        states = {id: [] for id in agents_id}
        rewards = {id: [] for id in agents_id}
        actions = {id: [] for id in agents_id}
        state = env.reset()
        while True:
            for id in agents_id:
                states[id].append(state.local[id])
            action = compute_action(policies, state)
            sim_begin = time.time()
            next_s, r, done, info = env.step(action)
            sim_cost += time.time() - sim_begin
            state = next_s

            for id in agents_id:
                rewards[id].append(r.local[id])
                actions[id].append(action.local[id])

            record.append_reward(r)
            record.append_info(info)

            if done.central:
                for id in agents_id:
                    traj = TrajectoryTuple(
                        states=states[id],
                        actions=actions[id],
                        rewards=rewards[id],
                        terminal=done.central,
                    )
                    samples_batch[id].trajectorys.append(traj)
                break
    learn_begin = time.time()
    policy_learn(policies=policies, samples=samples_batch)
    learn_end = time.time()
    learn_cost += (learn_end - learn_begin)

    logger.info("simulation time cost : {:.3f}s".format(sim_cost))
    logger.info("learning time cost : {:.3f}s".format(learn_cost))
    return record


def off_policy_train_fn(
    env: MultiAgentEnv,
    policies: Dict[str, Policy],
    buffers: Dict[str, ReplayBuffer],
    config: ExecutingConfig,
    logger: logging.Logger,
):
    batch_size = config["batch_size"]
    beta = config["per_beta"]
    sim_cost = 0.0
    learn_cost = 0.0

    state = env.reset()
    record = TrainingRecord()
    while True:
        action = compute_action(policies, state)

        sim_begin = time.time()
        next_s, r, done, info = env.step(action)
        sim_cost += time.time() - sim_begin

        replay_buffer_store(buffers, state, action, r, next_s, done)
        state = next_s

        learn_begin = time.time()
        sample_data = replay_buffer_sample(buffers, batch_size, beta)
        policy_learn(policies, sample_data)
        learn_cost += time.time() - learn_begin

        record.append_reward(r)
        record.append_info(info)

        if done.central:
            logger.info("simulation time cost : {:.3f}s".format(sim_cost))
            logger.info("learning time cost : {:.3f}s".format(learn_cost))
            break
    return record


def off_policy_per_train_fn(
    env: MultiAgentEnv,
    policies: Dict[str, Policy],
    buffers: Dict[str, ReplayBuffer],
    config: ExecutingConfig,
    logger: logging.Logger,
):
    batch_size = config["batch_size"]
    beta = config["per_beta"]

    sim_cost = 0.0
    learn_cost = 0.0

    state = env.reset()
    record = TrainingRecord()
    while True:
        action = compute_action(policies, state)

        sim_begin = time.time()
        next_s, r, done, info = env.step(action)
        sim_cost += time.time() - sim_begin
        replay_buffer_store(buffers, state, action, r, next_s, done)
        state = next_s

        learn_begin = time.time()
        sample_data = replay_buffer_sample(buffers, batch_size, beta)
        priorities = policy_learn(policies, sample_data)
        idxes = {}
        for id, element in sample_data.items():
            idxes[id] = element.idxes
        replay_buffer_update(buffers, idxes, priorities)
        learn_cost += time.time() - learn_begin

        record.append_reward(r)
        record.append_info(info)

        if done.central:
            logger.info("simulation time cost : {:.3f}s".format(sim_cost))
            logger.info("learning time cost : {:.3f}s".format(learn_cost))
            break
    return record


def compute_action(policies: Dict[str, Policy], state: State):
    actions = {}
    for id, p in policies.items():
        actions[id] = p.compute_action(state=state.local[id])
    action = Action(local=actions)
    return action


def replay_buffer_store(
    buffers: Dict[str, ReplayBuffer],
    state: State,
    action: Action,
    reward: Reward,
    next_state: State,
    terminal: Terminal,
):

    for id, buffer in buffers.items():
        trans = TransitionTuple(
            state=state.local[id],
            action=action.local[id],
            reward=reward.local[id],
            next_state=next_state.local[id],
            terminal=terminal.local[id],
        )
        buffer.store(trans)


def replay_buffer_sample(buffers, batch_size: int, beta: float):
    samples = {}
    for id, buffer in buffers.items():
        samples[id] = buffer.sample(batch_size, beta)
    return samples


def policy_learn(
    policies: Dict[str, Policy],
    samples: Dict[str, SampleBatch],
):
    priorities = {}
    for id, p in policies.items():
        info = p.learn_on_batch(samples[id])
        priorities[id] = info.get("priorities", [])
    return priorities


def replay_buffer_update(
    buffers: Dict[str, PrioritizedReplayBuffer],
    idxes: Dict[str, List],
    priorities: Dict[str, List],
):
    for id, buffer in buffers.items():
        buffer.update_priorities(idxes[id], priorities[id])


class IndependentLearnerTrainer(Trainer):
    def __init__(
        self,
        type: TrainnerTypes,
        env: MultiAgentEnv,
        policies: Dict[str, Policy],
        replay_buffers: Dict[str, ReplayBuffer],
        train_fn,
        recorder: Recorder,
        config: Dict,
    ):

        self.type = type
        self.env = env
        self.policies = policies
        self.replay_bufferes = replay_buffers
        self.train_fn = train_fn
        self.recorder = recorder
        self.config = config

        self.agents_id = self.env.get_agents_id()
        self.trained_iteration = 0

    def train(self, episodes: int):
        ckpt_frequency = self.config["ckpt_frequency"]
        init_beta = self.config.get("per_beta", 0)
        left_beta = 1.0 - init_beta
        for ep in range(episodes):
            logger.info("========== train episode {} begin ==========".format(
                self.trained_iteration))
            # beta increased by trained iteration
            # it will equal one in the end
            self.config["per_beta"] = (ep +
                                       1) / episodes * left_beta + init_beta
            record = self.train_fn(
                self.env,
                self.policies,
                self.replay_bufferes,
                self.config,
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

    def get_checkpoint(self):
        policy_weight = {}
        buffer_weight = {}
        for id, p in self.policies.items():
            policy_weight[id] = p.get_weight()
        for id, buffer in self.replay_bufferes.items():
            buffer_weight[id] = buffer.get_weight()
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
        for id, p in self.policies.items():
            p.set_weight(policy_w[id])
        for id, buffer in self.replay_bufferes.items():
            buffer.set_weight(buffer_w[id])
        self.recorder.add_records(records)

    def get_config(self):
        policy_config = {}
        for id, p in self.policies.items():
            policy_config[id] = p.get_config()
        buffer_config = {}
        for id, buffer in self.replay_bufferes.items():
            buffer_config[id] = buffer.get_config()
        config = {
            "type": self.type,
            "trained_iteration": self.trained_iteration,
            "policy": policy_config,
            "buffer": buffer_config,
            "executing": self.config,
        }
        return config

    def eval(self, episode: int):
        eval_polices = {}
        for id, p in self.policies.items():
            eval_polices[id] = p.unwrapped()

        for ep in range(episode):
            logger.info("+++++++++ eval episode {} begin +++++++++".format(ep))

            rewards = []
            infos = []

            state = self.env.reset()
            while True:
                action = compute_action(eval_polices, state)
                next_s, r, done, info = self.env.step(action)
                state = next_s

                rewards.append(r)
                infos.append(info)
                if done.central:
                    break
            record = TrainingRecord(ep, rewards, infos)
            self.recorder.print_record(record, logger, False)
            logger.info("+++++++++ eval episode {} end   +++++++++".format(ep))

    def get_records(self):
        self.recorder.get_records()

    def save_config(self, dir: str = ""):
        self.recorder.write_config(self.get_config(), dir)

    def save_records(self, dir: str = ""):
        self.recorder.write_records(dir)

    def save_checkpoint(self, dir: str = "", filename: str = ""):
        self.recorder.write_ckpt(self.get_checkpoint(), dir, filename)
