import abc
import time
import logging
import os
from typing import Dict, List
from hprl.trainer.trainer import Trainer
from hprl.replaybuffer import ReplayBuffer, PrioritizedReplayBuffer, ReplayBufferTypes
from hprl.policy import PolicyTypes, Policy
from hprl.recorder import Recorder
import hprl

from hprl.env import MultiAgentEnv
from hprl.util.typing import TrainingRecord, SampleBatch, TransitionTuple, TrajectoryTuple
from hprl.util.typing import State, Action, Reward, Terminal

logger = logging.getLogger(__name__)


class ILearnerTrainer(Trainer):
    """
    !!! This class can't be used directly !!!
    must implement _train function first 

    """
    def __init__(
        self,
        type: PolicyTypes,
        env: MultiAgentEnv,
        policies: Dict[str, Policy],
        config: Dict,
        trained_iteration=0,
        records: List[TrainingRecord] = [],
        recorder: Recorder = None,
        output_dir: str = "",
    ):
        if not recorder:
            recorder = hprl.recorder.DefaultRecorder()

        if output_dir and not os.path.exists(output_dir):
            raise ValueError(
                "output directory {} not exists".format(output_dir))

        self._type = type
        self._env = env
        self._policies = policies
        self._config = config

        self._records = records
        self._recorder = recorder
        self._output_dir = output_dir
        self._trained_iteration = trained_iteration
        self._agents_id = self._env.agents_id

    def train(
        self,
        episodes: int,
        ckpt_frequency: int = 0,
    ):
        for ep in range(1, episodes + 1):
            self._trained_iteration += 1
            logger.info(f"========== train episode {ep} begin ==========", )
            logger.info("total trained iteration : %d",
                        self._trained_iteration)
            # implement _train in subclass
            record = self._train(ep, episodes)
            record.set_episode(self._trained_iteration)

            self._recorder.log_record(record=record, logger=logger)
            self._records.append(record)
            if (ckpt_frequency > 0 and ep % ckpt_frequency == 0):
                self.save_checkpoint()
            logger.info("========= train end   =========")
        return self._records

    @abc.abstractmethod
    def _train(self, ep: int, episodes: int):
        ...

    def get_checkpoint(self):
        policy_weight = {}
        for id, p in self._policies.items():
            policy_weight[id] = p.get_weight()
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

    def load_checkpoint(self, checkpoint: Dict):
        weight = checkpoint["weight"]
        records = checkpoint["records"]
        policy_w = weight.get("policy")
        for id, p in self._policies.items():
            p.set_weight(policy_w[id])
        self._recorder.add_records(records)

    def get_config(self):
        policy_config = {}
        for id, p in self._policies.items():
            policy_config[id] = p.get_config()
        config = {
            "type": self._type,
            "trained_iteration": self._trained_iteration,
            "policy": policy_config,
            "executing": self._config,
        }
        return config

    def eval(self, episode: int):
        eval_polices = {}
        for id, p in self._policies.items():
            eval_polices[id] = p.unwrapped()

        for ep in range(episode):
            logger.info("+++++++++ eval episode {} begin +++++++++".format(ep))

            rewards = []
            infos = []

            state = self._env.reset()
            while True:
                action = compute_action(eval_polices, state)
                next_s, r, done, info = self._env.step(action)
                state = next_s

                rewards.append(r)
                infos.append(info)
                if done.central:
                    break
            record = TrainingRecord(ep, rewards, infos)
            self._recorder.print_record(record, logger, False)
            self._recorder.add_record(record)
            logger.info("+++++++++ eval episode {} end   +++++++++".format(ep))

    def get_records(self):
        self._records

    def save_config(self, dir: str = ""):
        self._recorder.write_config(self.get_config(), dir)

    def save_records(self, path=""):
        if not path:
            path = "records.json"
            if self._output_dir:
                path = f"{self._output_dir}/{path}"
        self._recorder.write_records(records=self._records, path=path)

    def save_checkpoint(self, path=""):
        ckpt = self.get_checkpoint()
        if not path:
            path = f"ckpt_{self._trained_iteration}.pth"
            if self._output_dir:
                path = f"{self._output_dir}/{path}"
        hprl.recorder.write_ckpt(ckpt=ckpt, path=path)


class IOffPolicyTrainer(ILearnerTrainer):
    def __init__(
        self,
        type: PolicyTypes,
        env: MultiAgentEnv,
        policies: Dict[str, Policy],
        buffers: Dict[str, ReplayBuffer],
        config: Dict,
        trained_iter=0,
        recorder: Recorder = None,
        output_dir="",
    ):
        super(IOffPolicyTrainer, self).__init__(
            type=type,
            env=env,
            policies=policies,
            recorder=recorder,
            config=config,
            trained_iteration=trained_iter,
            output_dir=output_dir,
        )
        self.buffers = buffers

    def _train(self, ep: int, episodes: int):
        batch_size = self._config["batch_size"]
        beta = 0

        buffer_type = self.buffers[list(self.buffers)[0]].type

        if buffer_type == ReplayBufferTypes.Prioritized:
            init_beta = self._config["per_beta"]
            beta = (ep + 1) / episodes * (1 - init_beta) + init_beta

        sim_cost = 0.0
        learn_cost = 0.0

        state = self._env.reset()
        record = TrainingRecord()
        while True:
            action = compute_action(self._policies, state)

            sim_begin = time.time()
            next_s, r, done, info = self._env.step(action)
            sim_cost += time.time() - sim_begin
            replay_buffer_store(self.buffers, state, action, r, next_s, done)
            state = next_s

            learn_begin = time.time()
            sample_data = replay_buffer_sample(self.buffers, batch_size, beta)
            priorities = policy_learn(self._policies, sample_data)
            if buffer_type == ReplayBufferTypes.Prioritized:
                idxes = {}
                for id, element in sample_data.items():
                    idxes[id] = element.idxes
                replay_buffer_update(self.buffers, idxes, priorities)
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

        buffer_weight = {}
        for id, buffer in self.buffers.items():
            buffer_weight[id] = buffer.get_weight()
        checkpoint["weight"]["buffer"] = buffer_weight
        return checkpoint

    def load_checkpoint(self, checkpoint: Dict):
        weight = checkpoint["weight"]
        buffer_w = weight["buffer"]
        for id, buffer in self.buffers.items():
            buffer.set_weight(buffer_w[id])

        super().load_checkpoint(checkpoint)

    def get_config(self):
        config = super().get_config()

        buffer_config = {}
        for id, buffer in self.buffers.items():
            buffer_config[id] = buffer.get_config()
        config["buffer"] = buffer_config
        return config


class IOnPolicyTrainer(ILearnerTrainer):
    def __init__(
        self,
        type: PolicyTypes,
        env: MultiAgentEnv,
        policies: Dict[str, Policy],
        recorder: Recorder,
        config: Dict,
    ):
        super(IOnPolicyTrainer, self).__init__(
            type=type,
            env=env,
            policies=policies,
            recorder=recorder,
            config=config,
        )

    def _train(self, ep: int, episodes: int):
        batch_size = self._config["batch_size"]
        sim_cost = 0.0
        learn_cost = 0.0

        agents_id = self._env.agents_id
        samples_batch = {id: SampleBatch() for id in agents_id}
        for _ in range(batch_size):
            record = TrainingRecord()
            states = {id: [] for id in agents_id}
            rewards = {id: [] for id in agents_id}
            actions = {id: [] for id in agents_id}
            state = self._env.reset()
            while True:
                for id in agents_id:
                    states[id].append(state.local[id])
                action = compute_action(self._policies, state)
                sim_begin = time.time()
                next_s, r, done, info = self._env.step(action)
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
        policy_learn(policies=self._policies, samples=samples_batch)
        learn_end = time.time()
        learn_cost += (learn_end - learn_begin)

        logger.info("simulation time cost : {:.3f}s".format(sim_cost))
        logger.info("learning time cost : {:.3f}s".format(learn_cost))
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