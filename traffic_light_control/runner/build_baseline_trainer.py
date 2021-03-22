from runner.build_model import build_model
from runner.build_env import build_env
from typing import Dict
import hprl


def build_baseline_trainer(
    env_id: str,
    policy_type: hprl.PolicyTypes,
    buffer_type: hprl.ReplayBufferTypes,
    batch_size: int = 0,
):
    if env_id not in ["1x1", "1x3"]:
        raise ValueError("env id invliad, only suppor 1x1 and 1x3 env")

    env = build_env(
        env_id=env_id,
        thread_num=1,
        save_replay=False,
    )
    models = build_model(policy=policy_type, env=env)

    config, episodes = _get_trainer_config(
        env_id=env_id,
        policy_type=policy_type,
        buffer_type=buffer_type,
        action_space=env.get_local_action_space(),
        state_space=env.get_local_state_space(),
    )

    if batch_size > 0:
        config["executing"]["batch_size"] = batch_size
    trainer = hprl.build_trainer(
        config=config,
        env=env,
        models=models,
    )
    return trainer, episodes


def _get_trainer_config(
    env_id: str,
    policy_type: hprl.PolicyTypes,
    buffer_type: hprl.ReplayBufferTypes,
    action_space,
    state_space,
):
    if policy_type == hprl.PolicyTypes.IQL:
        config = _get_iql_config(
            env_id=env_id,
            buffer_type=buffer_type,
            action_space=action_space,
            state_space=state_space,
        )
        episodes = 1000
    else:
        raise ValueError(" not support policy type for baseline")
    return config, episodes


def _get_iql_config(
    env_id: str,
    buffer_type: hprl.ReplayBufferTypes,
    action_space,
    state_space,
):
    print(env_id)
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