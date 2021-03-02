from hprl.policy.actor_critic import ActorCritic
import logging
from typing import Dict, List


from hprl.trainer.common_trainer import CommonTrainer
from hprl.util.enum import AdvantageTypes, ReplayBufferTypes, TrainnerTypes
from hprl.policy import Policy, DQN, ILearnerWrapper, EpsilonGreedy, PPO
from hprl.env import MultiAgentEnv
from hprl.replaybuffer import ReplayBuffer, CommonBuffer
from hprl.trainer.core import Trainer
from hprl.util.checkpointer import Checkpointer
from hprl.trainer.support_fn import default_log_record_fn, off_policy_train_fn, on_policy_train_fn

logger = logging.getLogger(__name__)


def create_trainer(
        config: Dict,
        env: MultiAgentEnv,
        models: Dict,
        replay_buffer: ReplayBuffer = None,
        policy: Policy = None,
        log_record_fn=None) -> Trainer:

    trainner_type = config["type"]
    buffer_config = config["buffer"]
    policy_config = config["policy"]
    executing_config = config["executing"]

    agents_id = env.get_agents_id()
    checkpoint_dir = executing_config["checkpoint_dir"]
    checkpoint_frequency = executing_config["check_frequency"]
    checkpointer = Checkpointer(
        base_directory=checkpoint_dir,
        checkpoint_frequency=checkpoint_frequency)

    if replay_buffer is None:
        replay_buffer_type = buffer_config["type"]
        replay_buffer = _create_replay_buffer(
            replay_buffer_type, buffer_config)

    if policy is None:
        policy = _create_policy(
            trainner_type, agents_id,
            policy_config, models)

    if log_record_fn is None:
        log_record_fn = default_log_record_fn

    train_fn = None

    if trainner_type == TrainnerTypes.IQL:
        train_fn = off_policy_train_fn
    elif (trainner_type == TrainnerTypes.PPO
          or trainner_type == TrainnerTypes.IAC):
        train_fn = on_policy_train_fn
    else:
        raise ValueError("trainner type {} is invalid".format(trainner_type))

    trainner = CommonTrainer(
        type=trainner_type,
        config=executing_config,
        train_fn=train_fn,
        env=env,
        policy=policy,
        replay_buffer=replay_buffer,
        checkpointer=checkpointer,
        log_record_fn=log_record_fn,
        record_base_dir=executing_config["record_base_dir"],
        cumulative_train_iteration=executing_config.get(
            "trained_iteration", 0)
    )
    return trainner


def load_trainer(
        env: MultiAgentEnv,
        models: Dict,
        checkpoint_dir: str,
        checkpoint_file: str):
    checkpoint = Checkpointer(checkpoint_dir)
    data = checkpoint.load(checkpoint_file)
    config = data.get("config")
    trainer = create_trainer(config, env, models)
    trainer.load_checkpoint(data)
    return trainer


def _create_replay_buffer(type, config):
    if type == ReplayBufferTypes.Common:
        capacity = config.get("capacity")
        return CommonBuffer(capacity)
    raise ValueError("replay buffer type {} is invalid".format(type))


def _create_policy(type, agents_id, config, models):
    if type == TrainnerTypes.IQL:
        policies = {}
        for id_ in agents_id:
            model = models[id_]
            inner_p = DQN(
                acting_net=model["acting"],
                target_net=model["target"],
                learning_rate=config["learning_rate"],
                discount_factor=config["discount_factor"],
                update_period=config["update_period"],
                action_space=config["action_space"],
                state_space=config["state_space"],
            )
            policies[id_] = EpsilonGreedy(
                inner_policy=inner_p,
                eps_frame=config["eps_frame"],
                eps_min=config["eps_min"],
                eps_init=config["eps_init"],
                action_space=config["action_space"],
            )
        i_learner = ILearnerWrapper(
            agents_id=agents_id,
            policies=policies,
        )
        return i_learner
    elif type == TrainnerTypes.PPO:
        policies = {}
        for id_ in agents_id:
            model = models[id_]
            policies[id_] = PPO(
                critic_net=model["critic_net"],
                critic_target_net=model["critic_target_net"],
                actor_net=model["actor_net"],
                inner_epoch=config["inner_epoch"],
                learning_rate=config["learning_rate"],
                discount_factor=config["discount_factor"],
                update_period=config["update_period"],
                action_space=config["action_space"],
                state_space=config["state_space"],
                clip_param=config["clip_param"],
                advantage_type=config["advantage_type"],
            )
        i_learner = ILearnerWrapper(
            agents_id=agents_id,
            policies=policies,
        )
        return i_learner
    elif type == TrainnerTypes.IAC:

        policies = {}
        for id_ in agents_id:
            model = models[id_]
            policies[id_] = ActorCritic(
                critic_net=model["critic_net"],
                critic_target_net=model["critic_target_net"],
                actor_net=model["actor_net"],
                learning_rate=config["learning_rate"],
                discount_factor=config["discount_factor"],
                update_period=config["update_period"],
                action_space=config["action_space"],
                state_space=config["state_space"],
                advantage_type=config["advantage_type"],
            )
        i_learner = ILearnerWrapper(
            agents_id=agents_id,
            policies=policies,
        )
        return i_learner
    raise ValueError(f'policy type {type} is invalid')
