from hprl.policy.decorator.epsilon_greedy import SingleEpsilonGreedy
from hprl.recorder.torch_recorder import TorchRecorder
from hprl.policy.single.per_dqn import PERDQN, build_iql_trainer
from hprl.replaybuffer.prioritized_replay_buffer import PrioritizedReplayBuffer
import logging
from typing import Dict, List

import torch.nn as nn

from hprl.env import MultiAgentEnv
from hprl.policy import Policy, DQN, PPO, ActorCritic
from hprl.policy import IndependentLearner, EpsilonGreedy
from hprl.policy import MultiAgentEpsilonGreedy
from hprl.policy import COMA, VDN
from hprl.replaybuffer import ReplayBuffer, CommonBuffer
from hprl.trainer import Trainer, CommonTrainer, IndependentLearnerTrainer
from hprl.util.checkpointer import Checkpointer
from hprl.util.enum import AdvantageTypes, ReplayBufferTypes, TrainnerTypes
from hprl.util.support_fn import off_policy_train_fn, on_policy_train_fn
from hprl.util.support_fn import default_log_record_fn

logger = logging.getLogger(__name__)


def build_trainer(config: Dict,
                  env: MultiAgentEnv,
                  models: Dict,
                  replay_buffer: ReplayBuffer = None,
                  policy: Policy = None,
                  log_record_fn=None) -> Trainer:

    trainer_type = config["type"]
    buffer_config = config["buffer"]
    policy_config = config["policy"]
    executing_config = config["executing"]
    if trainer_type == TrainnerTypes.IQL_PER:
        trainer = build_iql_trainer(config, env, models)
    else:
        if replay_buffer is None:
            replay_buffer = build_replay_buffer(buffer_config["type"],
                                                buffer_config)
        if policy is None:
            agents_id = env.get_agents_id()
            policy, train_fn = build_policy(trainer_type, agents_id,
                                            policy_config, models)

        if log_record_fn is None:
            log_record_fn = default_log_record_fn

        checkpointer = build_checkpointer(
            executing_config["checkpoint_dir"],
            executing_config["ckpt_frequency"],
        )

        trainer = CommonTrainer(
            type=trainer_type,
            config=executing_config,
            train_fn=train_fn,
            env=env,
            policy=policy,
            replay_buffer=replay_buffer,
            checkpointer=checkpointer,
            log_record_fn=log_record_fn,
            record_base_dir=executing_config["record_base_dir"],
            log_dir=executing_config["log_dir"],
            cumulative_train_iteration=executing_config.get(
                "trained_iteration", 1))
    return trainer


def load_trainer(
    env: MultiAgentEnv,
    models: Dict,
    checkpoint_dir: str,
    checkpoint_file: str,
):
    checkpoint = Checkpointer(checkpoint_dir)
    data = checkpoint.load(checkpoint_file)
    config = data.get("config")
    trainer_type = config["type"]
    if trainer_type == TrainnerTypes.IQL_PER:
        trainer = build_iql_trainer(config, env, models)
    else:
        trainer = build_trainer(config, env, models)
    trainer.load_checkpoint(data)
    return trainer


def build_checkpointer(dir: str, frequency: int):
    checkpoint_dir = dir
    checkpoint_frequency = frequency
    checkpointer = Checkpointer(base_directory=checkpoint_dir,
                                checkpoint_frequency=checkpoint_frequency)
    return checkpointer


def build_replay_buffer(type, config):
    if type == ReplayBufferTypes.Common:
        capacity = config["capacity"]
        return CommonBuffer(capacity)
    elif type == ReplayBufferTypes.Prioritized:
        capacity = config["capacity"]
        alpha = config["alpha"]
        return PrioritizedReplayBuffer(capacity, alpha)
    raise ValueError("replay buffer type {} is invalid".format(type))


def build_policy(type, agents_id, config, models):
    if (type == TrainnerTypes.IQL or type == TrainnerTypes.IQL_PS):
        return _create_iql(config, models, agents_id)
    elif (type == TrainnerTypes.PPO or type == TrainnerTypes.PPO_PS):
        return _create_ppo(config, models, agents_id)
    elif type == TrainnerTypes.IAC:
        return _create_iac(config, models, agents_id)
    elif type == TrainnerTypes.VDN:
        return _create_vdn(config, models, agents_id)
    elif type == TrainnerTypes.COMA:
        return _create_coma(config, models, agents_id)
    raise ValueError(f'policy type {type} is invalid')


def _create_iql(config, models, agents_id):
    policies = {}
    for id_ in agents_id:
        model = models[id_]
        inner_p = DQN(
            acting_net=model["acting_net"],
            target_net=model["target_net"],
            critic_lr=config["critic_lr"],
            discount_factor=config["discount_factor"],
            update_period=config["update_period"],
            action_space=config["action_space"][id_],
            state_space=config["state_space"][id_],
        )
        policies[id_] = EpsilonGreedy(
            inner_policy=inner_p,
            eps_frame=config["eps_frame"],
            eps_min=config["eps_min"],
            eps_init=config["eps_init"],
            action_space=config["action_space"][id_],
        )
    i_learner = IndependentLearner(
        agents_id=agents_id,
        policies=policies,
    )
    return i_learner, off_policy_train_fn


def _create_ppo(config, models, agents_id):
    policies = {}
    for id_ in agents_id:
        model = models[id_]
        policies[id_] = PPO(
            critic_net=model["critic_net"],
            critic_target_net=model["critic_target_net"],
            actor_net=model["actor_net"],
            inner_epoch=config["inner_epoch"],
            critic_lr=config["critic_lr"],
            actor_lr=config["actor_lr"],
            discount_factor=config["disount_factor"],
            update_period=config["update_period"],
            action_space=config["action_space"],
            state_space=config["state_space"],
            clip_param=config["clip_param"],
            advantage_type=config["advantage_type"],
        )
    i_learner = IndependentLearner(
        agents_id=agents_id,
        policies=policies,
    )
    return i_learner, on_policy_train_fn


def _create_iac(config, models, agents_id):
    policies = {}
    for id_ in agents_id:
        model = models[id_]
        policies[id_] = ActorCritic(
            critic_net=model["critic_net"],
            critic_target_net=model["critic_target_net"],
            actor_net=model["actor_net"],
            critic_lr=config["critic_lr"],
            actor_lr=config["actor_lr"],
            discount_factor=config["discount_factor"],
            update_period=config["update_period"],
            action_space=config["action_space"],
            state_space=config["state_space"],
            advantage_type=config["advantage_type"],
        )
    i_learner = IndependentLearner(
        agents_id=agents_id,
        policies=policies,
    )
    return i_learner, on_policy_train_fn


def _create_vdn(config, models, agents_id):
    inner_p = VDN(
        agents_id=agents_id,
        acting_nets=models["acting_nets"],
        target_nets=models["target_nets"],
        critic_lr=config["critic_lr"],
        discount_factor=config["discount_factor"],
        update_period=config["update_period"],
        action_space=config["action_space"],
        state_space=config["state_space"],
    )
    p = MultiAgentEpsilonGreedy(
        agents_id=agents_id,
        inner_policy=inner_p,
        eps_frame=config["eps_frame"],
        eps_min=config["eps_min"],
        eps_init=config["eps_init"],
        action_space=config["action_space"],
    )
    return p, off_policy_train_fn


def _create_coma(config, models, agents_id):
    p = COMA(
        agents_id=agents_id,
        critic_net=models["critic_net"],
        critic_target_net=models["target_critic_net"],
        actors_net=models["actors_net"],
        critic_lr=config["critic_lr"],
        actor_lr=config["actor_lr"],
        discount_factor=config["discount_factor"],
        update_period=config["update_period"],
        clip_param=config["clip_param"],
        inner_epoch=config["inner_epoch"],
        local_action_space=config["action_space"],
        local_state_space=config["state_space"],
    )
    return p, on_policy_train_fn
