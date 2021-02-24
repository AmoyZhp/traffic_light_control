from hprl.policy.epsilon_greedy import EpsilonGreedy
from hprl.util.checkpointer import Checkpointer
from typing import Dict, List

import torch
import torch.nn as nn

from hprl.util.enum import ReplayBufferTypes, TrainnerTypes
from hprl.util.typing import Reward, Transition
from hprl.policy import Policy, DQN, ILearnerWrapper
from hprl.env import MultiAgentEnv
from hprl.replaybuffer import ReplayBuffer, CommonBuffer
from hprl.trainer.qlearning_trainer import QLearningTranier
from hprl.trainer.core import Trainer


def _off_policy_train_fn(
        env: MultiAgentEnv,
        policy: Policy,
        replay_buffer: ReplayBuffer,
        config: Dict) -> List[Reward]:

    batch_size = config["batch_size"]

    state = env.reset()
    reward_record = []
    while True:
        action = policy.compute_action(state)
        ns, r, done, _ = env.step(action)

        trans = Transition(
            state=state,
            action=action,
            reward=r,
            next_state=ns,
            terminal=done,
        )
        replay_buffer.store(trans)

        state = ns
        reward_record.append(r)

        batch_data = replay_buffer.sample(batch_size)
        policy.learn_on_batch(batch_data)
        if done.central:
            break
    return reward_record


def _on_policy_train_fn(
        env: MultiAgentEnv,
        policy: Policy,
        replay_buffer: ReplayBuffer,
        config: Dict):

    batch_size = config["batch_size"]
    reward_record = []

    state = env.reset()

    while True:
        action = policy.compute_action(state)
        ns, r, done, _ = env.step(action)

        trans = Transition(
            state=state,
            action=action,
            reward=r,
            next_state=ns,
            terminal=done,
        )
        replay_buffer.store(trans)

        state = ns
        reward_record.append(r)

        if done:
            batch_data = replay_buffer.sample(batch_size)
            policy.learn_on_batch(batch_data)
            break


def load_trainer(
        env: MultiAgentEnv,
        models: Dict,
        checkpoint_dir: str,
        checkpoint_file: str):
    checkpoint = Checkpointer(checkpoint_dir)
    data = checkpoint.load(checkpoint_file)
    config = data.get("config")
    weight = data.get("weight")
    trainer = create_trainer(config, env, models)
    trainer.set_weight(weight)
    return trainer


def create_trainer(
        config: Dict,
        env: MultiAgentEnv,
        models: Dict,
        replay_buffer: ReplayBuffer = None,
        policy: Policy = None) -> Trainer:

    trainner_type = config.get("type")
    buffer_config = config.get("buffer")
    policy_config = config.get("policy")
    executing_config = config.get("executing")

    agents_id = env.get_agents_id()
    base_directory = executing_config.get("base_dir")
    checkpoint_frequency = executing_config.get("check_frequency")
    checkpointer = Checkpointer(
        base_directory=base_directory,
        checkpoint_frequency=checkpoint_frequency)

    if trainner_type == TrainnerTypes.IQL:

        if replay_buffer is None:
            replay_buffer_type = buffer_config.get("type")
            replay_buffer = _create_replay_buffer(
                replay_buffer_type, buffer_config)

        if policy is None:
            policy = _create_policy(
                trainner_type, agents_id,
                policy_config, models)
        trainner = QLearningTranier(
            type=trainner_type,
            config=executing_config,
            train_fn=_off_policy_train_fn,
            env=env,
            policy=policy,
            replay_buffer=replay_buffer,
            checkpointer=checkpointer,
            cumulative_train_iteration=executing_config.get(
                "trained_iteration", 0)
        )
        return trainner

    raise ValueError("trainner type {} is invalid".format(trainner_type))


def _create_replay_buffer(type, config):
    if type == ReplayBufferTypes.Common:
        capacity = config.get("capacity")
        return CommonBuffer(capacity)
    raise ValueError("replay buffer type {} is invalid".format(type))


def _create_policy(type, agents_id, config, models):
    policies = {}
    if type == TrainnerTypes.IQL:
        for id_ in agents_id:
            model = models[id_]

            inner_p = DQN(
                acting_net=model.get("acting"),
                target_net=model.get("target"),
                learning_rate=config.get("learning_rate"),
                discount_factor=config.get("discount_factor"),
                update_period=config.get("update_period"),
                action_space=config.get("action_space"),
                state_space=config.get("state_space"),
            )
            policies[id_] = EpsilonGreedy(
                inner_policy=inner_p,
                eps_frame=config.get("eps_frame"),
                eps_min=config.get("eps_min"),
                eps_init=config.get("eps_init"),
                action_space=config.get("action_space"),
            )
        i_learner = ILearnerWrapper(
            agents_id=agents_id,
            policies=policies,
        )
        return i_learner
    raise ValueError(f'policy type {type} is invalid')
