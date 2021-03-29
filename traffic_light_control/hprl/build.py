from hprl.recorder.recorder import Recorder
import gym
from hprl.env.gym_wrapper import GymWrapper
import hprl.policy.dqn as dqn
import hprl.policy.vdn as vdn
import hprl.policy.ac as ac
import hprl.policy.coma as coma
import hprl.policy.qmix as qmix
import logging
from typing import Dict, List
from hprl.trainer.trainer import Trainer
from hprl.env import MultiAgentEnv
from hprl.policy import PolicyTypes
from hprl.replaybuffer import ReplayBufferTypes

logger = logging.getLogger(__package__)


def build_trainer(
    config: Dict,
    env: MultiAgentEnv,
    models: Dict,
    recorder: Recorder = None,
) -> Trainer:

    trainer_type = config["type"]
    trainer = None
    if trainer_type == PolicyTypes.IQL:
        trainer = dqn.build_iql_trainer(config, env, models, recorder)
    elif trainer_type == PolicyTypes.IAC:
        trainer = ac.build_iac_trainer(config, env, models)
    elif trainer_type == PolicyTypes.PPO:
        trainer = ac.build_ppo_trainer(config, env, models)
    elif trainer_type == PolicyTypes.VDN:
        trainer = vdn.build_vdn_trainer(config, env, models, recorder)
    elif trainer_type == PolicyTypes.COMA:
        trainer = coma.build_coma_trainer(config, env, models)
    elif trainer_type == PolicyTypes.QMIX:
        trainer = qmix.build_qmix_trainer(config, env, models, recorder)
    else:
        raise ValueError("train type %s is invalid", trainer_type)

    return trainer


def gym_baseline_trainer(
    trainer_type: PolicyTypes,
    buffer_type: ReplayBufferTypes = None,
    batch_size: int = 0,
) -> Trainer:
    if trainer_type == PolicyTypes.IQL:
        if buffer_type is None:
            raise ValueError("buffre type could be None for IQL")
        config, model = dqn.get_test_setting(buffer_type)
        env = GymWrapper(gym.make("CartPole-v1"))
        id = env.agents_id[0]
        models = {id: model}
        config["policy"]["action_space"][id] = 2
        config["policy"]["state_space"][id] = 4
        if batch_size > 0:
            config["executing"]["batch_size"] = batch_size

        trainer = dqn.build_iql_trainer(
            config=config,
            env=env,
            models=models,
        )
        return trainer
    elif trainer_type == PolicyTypes.IAC:
        config, model = ac.get_ac_test_setting()
        env = GymWrapper(gym.make("CartPole-v1"))
        id = env.agents_id[0]
        models = {id: model}
        config["policy"]["action_space"][id] = 2
        config["policy"]["state_space"][id] = 4
        if batch_size > 0:
            config["executing"]["batch_size"] = batch_size
        trainer = ac.build_iac_trainer(
            config=config,
            env=env,
            models=models,
        )
        return trainer
    elif trainer_type == PolicyTypes.PPO:
        config, model = ac.get_ppo_test_setting()
        env = GymWrapper(gym.make("CartPole-v1"))
        id = env.agents_id[0]
        models = {id: model}
        config["policy"]["action_space"][id] = 2
        config["policy"]["state_space"][id] = 4
        if batch_size > 0:
            config["executing"]["batch_size"] = batch_size
        trainer = ac.build_ppo_trainer(
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
