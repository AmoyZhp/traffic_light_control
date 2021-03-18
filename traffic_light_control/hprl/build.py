from hprl.old_build import old_build_trainer
import gym
from hprl.env.gym_wrapper import GymWrapper
import hprl.policy.dqn as dqn
import hprl.policy.vdn as vdn
import hprl.policy.ac as ac
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
    elif trainer_type == TrainnerTypes.IAC:
        trainer = ac.build_iac_trainer(config, env, models)
    elif trainer_type == TrainnerTypes.PPO:
        trainer = ac.build_ppo_trainer(config, env, models)
    elif trainer_type == TrainnerTypes.VDN:
        trainer = vdn.build_vdn_trainer(config, env, models)
    else:
        trainer = old_build_trainer(
            config=config,
            env=env,
            models=models,
        )
    return trainer


def gym_baseline_trainer(
    trainer_type: TrainnerTypes,
    buffer_type: ReplayBufferTypes = None,
) -> Trainer:
    if trainer_type == TrainnerTypes.IQL:
        if buffer_type is None:
            raise ValueError("buffre type could be None for IQL")
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
    elif trainer_type == TrainnerTypes.IAC:
        config, model = ac.get_ac_test_setting()
        env = GymWrapper(gym.make("CartPole-v1"))
        id = env.get_agents_id()[0]
        models = {id: model}
        config["policy"]["action_space"][id] = 2
        config["policy"]["state_space"][id] = 4
        trainer = ac.build_ac_trainer(
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
