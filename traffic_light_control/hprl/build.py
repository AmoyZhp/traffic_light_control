from hprl.old_build import old_build_trainer
import gym
from hprl.env.gym_wrapper import GymWrapper
import hprl.policy.dqn as dqn
import logging
from typing import Dict, List
from hprl.trainer.trainer import Trainer
from hprl.env import MultiAgentEnv
from hprl.util.checkpointer import Checkpointer
from hprl.util.enum import ReplayBufferTypes, TrainnerTypes

logger = logging.getLogger(__package__)


def build_trainer(
    config: Dict,
    env: MultiAgentEnv,
    models: Dict,
) -> Trainer:

    trainer_type = config["type"]
    trainer = None
    if trainer_type == TrainnerTypes.IQL:
        trainer = dqn.build_iql_trainer(config, env, models)
    elif trainer_type == TrainnerTypes.PPO:
        ...
    elif trainer_type == TrainnerTypes.VDN:
        ...
    else:
        trainer = old_build_trainer(
            config=config,
            env=env,
            models=models,
        )
    return trainer


def test_trainer(
    trainer_type: TrainnerTypes,
    buffer_type: ReplayBufferTypes = None,
) -> Trainer:
    if trainer_type == TrainnerTypes.IQL:
        if buffer_type is None:
            raise ValueError("buffre type could be None")
        config, model = dqn.get_test_setting(buffer_type)
        env = GymWrapper(gym.make("CartPole-v1"))
        id = env.get_agents_id()[0]
        models = {id: model}
        config["policy"]["action_space"][id] = 2
        config["policy"]["state_space"][id] = 4
        trainer = dqn.build_iql_trainer(
            config=config,
            env=env,
            models=models,
        )
        return trainer
    else:
        raise ValueError("trainer type invalid {}".format(trainer_type))


def load_trainer(
    env: MultiAgentEnv,
    models: Dict,
    checkpoint_dir: str,
    checkpoint_file: str,
):
    ...
