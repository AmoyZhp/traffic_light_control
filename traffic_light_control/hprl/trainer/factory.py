import logging
from typing import Dict, List


from hprl.trainer.common_trainer import CommonTrainer
from hprl.util.enum import ReplayBufferTypes, TrainnerTypes
from hprl.policy import Policy, DQN, ILearnerWrapper, EpsilonGreedy
from hprl.env import MultiAgentEnv
from hprl.replaybuffer import ReplayBuffer, CommonBuffer
from hprl.trainer.core import Trainer
from hprl.util.checkpointer import Checkpointer
from hprl.trainer.support_fn import default_log_record_fn, off_policy_train_fn

logger = logging.getLogger(__name__)


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
    base_directory = executing_config["base_dir"]
    checkpoint_frequency = executing_config["check_frequency"]
    checkpointer = Checkpointer(
        base_directory=base_directory,
        checkpoint_frequency=checkpoint_frequency)

    if trainner_type == TrainnerTypes.IQL:

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

        trainner = CommonTrainer(
            type=trainner_type,
            config=executing_config,
            train_fn=off_policy_train_fn,
            env=env,
            policy=policy,
            replay_buffer=replay_buffer,
            checkpointer=checkpointer,
            log_record_fn=log_record_fn,
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
    raise ValueError(f'policy type {type} is invalid')
