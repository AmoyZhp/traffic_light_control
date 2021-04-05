import os
from hprl.policy.dqn.build import build_iql_trainer
from hprl.recorder.recorder import Recorder, read_ckpt
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
from hprl import log_to_file
import collections.abc

logger = logging.getLogger(__name__)


def dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def build_trainer(
    config: Dict,
    env: MultiAgentEnv,
    models: Dict,
    recorder: Recorder = None,
    load=False,
) -> Trainer:

    trainer_type = config["type"]
    trainer = None
    if trainer_type == PolicyTypes.IQL:
        trainer = dqn.build_iql_trainer(
            config,
            env,
            models,
            recorder,
            load=load,
        )
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
    ckpt_path: str,
    override_config: Dict = {},
    recording: bool = False,
):
    ckpt = read_ckpt(ckpt_path)
    config: Dict = ckpt["config"]
    trainer_conf = config["trainer"]
    policy_type = trainer_conf["type"]

    if recording:
        old_output_dir: str = trainer_conf["output_dir"]
        split_str = old_output_dir.split("_")
        cnt = 0
        if split_str[-2] == "cnt":
            cnt = int(split_str[-1])
            preffix = old_output_dir.split("cnt")[0]
            output_dir = f"{preffix}cnt_{cnt+1}"
        else:
            output_dir = f"{old_output_dir}_cnt_0"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        log_to_file(output_dir)
    else:
        output_dir = ""

    if "trainer" not in override_config:
        override_config["trainer"] = {}
    override_config["trainer"]["output_dir"] = output_dir
    if policy_type == PolicyTypes.IQL:
        dict_update(config, override_config)
        trainer = build_iql_trainer(config)
        trainer.load_checkpoint(ckpt)
    else:
        raise ValueError(
            "policy {} loading function not implement".format(policy_type))
    return trainer


def _load_trainer(
    env: MultiAgentEnv,
    models: Dict,
    ckpt: Dict,
) -> Trainer:

    config = ckpt["config"]
    trainer = build_trainer(config=config, env=env, models=models, load=True)
    trainer.load_checkpoint(ckpt)
    return trainer
