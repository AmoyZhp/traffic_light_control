from typing import Dict

import hprl

from runner.build_model import build_model


def build_baseline_trainer(
    env_id: str,
    policy_type: hprl.PolicyTypes,
    buffer_type: hprl.ReplayBufferTypes,
    batch_size: int = 0,
):
    if env_id not in ["1x1", "1x3"]:
        raise ValueError("env id invliad, only suppor 1x1 and 1x3 env")
    raise NotImplementedError


def _get_trainer_config(
    env_id: str,
    policy_type: hprl.PolicyTypes,
    buffer_type: hprl.ReplayBufferTypes,
    action_space,
    state_space,
):
    if policy_type == hprl.PolicyTypes.IQL:
        episodes = 1000
        config = _get_iql_config(
            env_id=env_id,
            buffer_type=buffer_type,
            action_space=action_space,
            state_space=state_space,
        )
    elif policy_type == hprl.PolicyTypes.PPO:
        episodes = 600
        config = _get_ppo_config(
            env_id=env_id,
            action_space=action_space,
            state_space=state_space,
        )
    elif policy_type == hprl.PolicyTypes.IAC:
        episodes = 1000
        config = _get_iac_config(
            env_id=env_id,
            action_space=action_space,
            state_space=state_space,
        )
    elif policy_type == hprl.PolicyTypes.VDN:
        episodes = 1000
        config = _get_vdn_config(
            env_id=env_id,
            buffer_type=buffer_type,
            action_space=action_space,
            state_space=state_space,
        )
    elif policy_type == hprl.PolicyTypes.QMIX:
        episodes = 1000
        config = _get_qmix_config(
            env_id=env_id,
            buffer_type=buffer_type,
            action_space=action_space,
            state_space=state_space,
        )
    else:
        raise ValueError(" not support policy type for baseline")
    return config, episodes


def _get_qmix_config(
    env_id: str,
    buffer_type: hprl.ReplayBufferTypes,
    action_space,
    state_space,
):
    if env_id == "1x3":
        capacity = 20000
        critic_lr = 1e-4
        batch_size = 256
        discount_factor = 0.99
        eps_init = 1.0
        eps_min = 0.0001
        eps_frame = 300000
        update_period = 500
        action_space = action_space
        state_space = state_space
        policy_config = {
            "critic_lr": critic_lr,
            "discount_factor": discount_factor,
            "update_period": update_period,
            "action_space": action_space,
            "state_space": state_space,
            "eps_frame": eps_frame,
            "eps_init": eps_init,
            "eps_min": eps_min,
        }
        buffer_config = {
            "type": buffer_type,
            "capacity": capacity,
        }
        exec_config = {
            "batch_size": batch_size,
            "recording": False,
        }
        if buffer_type == hprl.ReplayBufferTypes.Prioritized:
            buffer_config["alpha"] = 0.4
            exec_config["per_beta"] = 0.6

        trainner_config = {
            "type": hprl.PolicyTypes.QMIX,
            "executing": exec_config,
            "policy": policy_config,
            "buffer": buffer_config,
        }
        return trainner_config
    raise ValueError("QMIX baseline config only support env of 1x3")


def _get_vdn_config(
    env_id: str,
    buffer_type: hprl.ReplayBufferTypes,
    action_space,
    state_space,
):
    if env_id == "1x3":
        capacity = 20000
        critic_lr = 1e-4
        batch_size = 256
        discount_factor = 0.99
        eps_init = 1.0
        eps_min = 0.0001
        eps_frame = 300000
        update_period = 500
        action_space = action_space
        state_space = state_space
        policy_config = {
            "critic_lr": critic_lr,
            "discount_factor": discount_factor,
            "update_period": update_period,
            "action_space": action_space,
            "state_space": state_space,
            "eps_frame": eps_frame,
            "eps_init": eps_init,
            "eps_min": eps_min,
        }
        buffer_config = {
            "type": buffer_type,
            "capacity": capacity,
        }
        exec_config = {
            "batch_size": batch_size,
            "recording": False,
        }
        if buffer_type == hprl.ReplayBufferTypes.Prioritized:
            buffer_config["alpha"] = 0.4
            exec_config["per_beta"] = 0.6

        trainner_config = {
            "type": hprl.PolicyTypes.VDN,
            "executing": exec_config,
            "policy": policy_config,
            "buffer": buffer_config,
        }
        return trainner_config
    raise ValueError("VDN baseline config only support env of 1x3")


def _get_iac_config(
    env_id: str,
    action_space,
    state_space,
):
    if env_id == "1x1" or env_id == "1x3":
        critic_lr = 1e-4
        actor_lr = 1e-4
        batch_size = 32
        discount_factor = 0.99
        update_period = 500
        action_space = action_space
        state_space = state_space
        advg_type = hprl.AdvantageTypes.QMinusV
        policy_config = {
            "critic_lr": critic_lr,
            "actor_lr": actor_lr,
            "discount_factor": discount_factor,
            "update_period": update_period,
            "action_space": action_space,
            "state_space": state_space,
            "advg_type": advg_type,
        }
        exec_config = {
            "batch_size": batch_size,
            "recording": False,
        }
        trainner_config = {
            "type": hprl.PolicyTypes.IAC,
            "executing": exec_config,
            "policy": policy_config,
        }
        return trainner_config

    raise ValueError("IAC baseline config only support env of 1x1 and 1x3")


def _get_ppo_config(
    env_id: str,
    action_space,
    state_space,
):
    if env_id == "1x1" or env_id == "1x3":
        critic_lr = 1e-4
        actor_lr = 1e-4
        batch_size = 32
        discount_factor = 0.99
        update_period = 500
        inner_epoch = 128
        clip_param = 0.2
        action_space = action_space
        state_space = state_space
        advg_type = hprl.AdvantageTypes.QMinusV
        policy_config = {
            "critic_lr": critic_lr,
            "actor_lr": actor_lr,
            "discount_factor": discount_factor,
            "update_period": update_period,
            "action_space": action_space,
            "state_space": state_space,
            "inner_epoch": inner_epoch,
            "clip_param": clip_param,
            "advg_type": advg_type,
        }
        exec_config = {
            "batch_size": batch_size,
            "recording": False,
        }
        trainner_config = {
            "type": hprl.PolicyTypes.PPO,
            "executing": exec_config,
            "policy": policy_config,
        }
        return trainner_config

    raise ValueError("PPO baseline config only support env of 1x1 and 1x3")


def _get_iql_config(
    env_id: str,
    buffer_type: hprl.ReplayBufferTypes,
    action_space,
    state_space,
):
    if env_id == "1x1" or env_id == "1x3":
        capacity = 20000
        critic_lr = 1e-4
        batch_size = 256
        discount_factor = 0.99
        eps_init = 1.0
        eps_min = 0.0001
        eps_frame = 300000
        update_period = 500
        action_space = action_space
        state_space = state_space
        policy_config = {
            "critic_lr": critic_lr,
            "discount_factor": discount_factor,
            "update_period": update_period,
            "action_space": action_space,
            "state_space": state_space,
            "eps_frame": eps_frame,
            "eps_init": eps_init,
            "eps_min": eps_min,
        }
        buffer_config = {
            "type": buffer_type,
            "capacity": capacity,
        }
        exec_config = {
            "batch_size": batch_size,
            "recording": False,
        }
        if buffer_type == hprl.ReplayBufferTypes.Prioritized:
            buffer_config["alpha"] = 0.4
            exec_config["per_beta"] = 0.6

        trainner_config = {
            "type": hprl.PolicyTypes.IQL,
            "episodes": 1000,
            "executing": exec_config,
            "policy": policy_config,
            "buffer": buffer_config,
        }
        return trainner_config
    raise ValueError("IQL baseline config only support env of 1x1 and 1x3")
