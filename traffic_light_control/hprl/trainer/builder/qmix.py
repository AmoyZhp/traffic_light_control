import logging
from typing import Dict

import torch
import torch.nn as nn
from hprl.env import MultiAgentEnv
from hprl.env import make as make_env
from hprl.policy.model_registration import make_model
from hprl.policy.multi.qmix import QMIX
from hprl.policy.policy import PolicyTypes
from hprl.policy.wrapper import MAEpsilonGreedy
from hprl.replaybuffer import ReplayBufferTypes, build_basis, build_per
from hprl.trainer.basis import OffPolicyTrainer

logger = logging.getLogger(__name__)


def build_qmix_trainer(config: Dict):
    logger.info("start to create VDN traienr")
    policy_config = config["policy"]
    env_setting = config["env"]
    env_id = env_setting["type"]
    env: MultiAgentEnv = make_env(id=env_id, config=env_setting)
    agents_id = env.agents_id

    model_id = policy_config["model_id"]
    model_config = {
        "central_state": env.central_state_space,
        "local_state": env.local_state_space,
        "central_action": env.central_action_space,
        "local_action": env.local_action_space,
    }
    models: Dict = make_model(
        id=model_id,
        config=model_config,
        agents_id=agents_id,
    )
    buffer_config = config["buffer"]
    buffer_type = buffer_config["type"]
    prioritized = False
    loss_fn = nn.MSELoss()
    if buffer_type is ReplayBufferTypes.Prioritized:
        prioritized = True
        loss_fn = _per_loss_fn
    if not prioritized:
        buffer = build_basis(buffer_config, multi=True)
    else:
        buffer = build_per(buffer_config, mulit=True)

    critic_lr = policy_config["critic_lr"]
    discount_factor = policy_config["discount_factor"]
    update_period = policy_config["update_period"]
    eps_frame = policy_config["eps_frame"]
    eps_min = policy_config["eps_min"]
    eps_init = policy_config["eps_init"]

    local_action_space = env.local_action_space
    local_state_space = env.local_state_space
    central_action_space = env.central_action_space
    central_state_space = env.central_state_space

    inner_p = QMIX(
        agents_id=agents_id,
        actors_net=models["actors_net"],
        actors_target_net=models["actors_target_net"],
        critic_net=models["critic_net"],
        critic_target_net=models["critic_target_net"],
        critic_lr=critic_lr,
        discount_factor=discount_factor,
        update_period=update_period,
        local_action_space=local_action_space,
        local_state_space=local_state_space,
        central_state_space=central_state_space,
        central_action_space=central_action_space,
        loss_fn=loss_fn,
        prioritized=prioritized,
    )
    policy = MAEpsilonGreedy(
        agents_id=agents_id,
        inner_policy=inner_p,
        eps_frame=eps_frame,
        eps_min=eps_min,
        eps_init=eps_init,
        action_space=local_action_space,
    )
    training_config = config["trainer"]
    trained_iter = training_config.get("trained_iteration", 0)
    output_dir = training_config.get("output_dir", "")
    trainer = OffPolicyTrainer(
        type=PolicyTypes.IQL,
        policy=policy,
        buffer=buffer,
        env=env,
        config=training_config,
        trained_iteration=trained_iter,
        output_dir=output_dir,
    )
    return trainer


def _per_loss_fn(input, target, weight):
    return torch.mean((input - target)**2 * weight)
