import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hprl.env import MultiAgentEnv
from hprl.env import make as make_env
from hprl.policy.dqn.dqn import DQN
from hprl.policy.model_registration import make_model
from hprl.policy.policy import PolicyTypes
from hprl.policy.wrapper import EpsilonGreedy
from hprl.policy.wrapper import IndependentWrapper as PolicyWrapper
from hprl.replaybuffer import ReplayBufferTypes, build_basis, build_per
from hprl.replaybuffer.wrapper import IndependentWrapper as BufferWrapper
from hprl.trainer import OffPolicyTrainer

logger = logging.getLogger(__name__)


def build_trainer(config: Dict):
    logger.info("start to create IQL trainer")
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

    critic_lr = policy_config["critic_lr"]
    discount_factor = policy_config["discount_factor"]
    update_period = policy_config["update_period"]
    eps_frame = policy_config["eps_frame"]
    eps_min = policy_config["eps_min"]
    eps_init = policy_config["eps_init"]

    buffers = {}
    policies = {}
    for id in agents_id:
        logger.info("init agent %s", id)
        if not prioritized:
            buffers[id] = build_basis(buffer_config, multi=False)
        else:
            buffers[id] = build_per(buffer_config, mulit=False)
        model = models[id]
        action_space = policy_config["local_action_space"][id]
        state_space = policy_config["local_state_space"][id]
        inner_p = DQN(
            acting_net=model["acting_net"],
            target_net=model["target_net"],
            learning_rate=critic_lr,
            discount_factor=discount_factor,
            update_period=update_period,
            action_space=action_space,
            state_space=state_space,
            prioritized=prioritized,
            loss_fn=loss_fn,
        )
        policies[id] = EpsilonGreedy(
            inner_policy=inner_p,
            eps_frame=eps_frame,
            eps_min=eps_min,
            eps_init=eps_init,
            action_space=action_space,
        )

    training_config = config["trainer"]
    trained_iter = training_config.get("trained_iteration", 0)
    output_dir = training_config.get("output_dir", "")

    policy = PolicyWrapper(
        type=PolicyTypes.IQL,
        policies=policies,
    )

    buffer = BufferWrapper(type=buffer_type, buffers=buffers)

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


def get_test_setting(buffer_type: ReplayBufferTypes):
    capacity = 4000
    critic_lr = 1e-3
    batch_size = 64
    discount_factor = 0.99
    eps_init = 1.0
    eps_min = 0.01
    eps_frame = 5000
    update_period = 100
    action_space = 2
    state_space = 4
    alpha = 0.6
    beta = 0.4
    policy_config = {
        "critic_lr": critic_lr,
        "discount_factor": discount_factor,
        "update_period": update_period,
        "action_space": {},
        "state_space": {},
        "eps_frame": eps_frame,
        "eps_init": eps_init,
        "eps_min": eps_min,
    }
    buffer_config = {
        "type": buffer_type,
        "capacity": capacity,
        "alpha": alpha,
    }
    exec_config = {
        "batch_size": batch_size,
        "per_beta": beta,
        "recording": True,
        "ckpt_frequency": 0,
        "record_base_dir": "gym_test",
    }
    trainner_config = {
        "type": PolicyTypes.IQL,
        "executing": exec_config,
        "policy": policy_config,
        "buffer": buffer_config,
    }
    acting_net = CartPole(input_space=state_space, output_space=action_space)

    target_net = CartPole(input_space=state_space, output_space=action_space)

    model = {
        "acting_net": acting_net,
        "target_net": target_net,
    }

    return trainner_config, model


def _per_loss_fn(input, target, weight):
    return torch.mean((input - target)**2 * weight)


class CartPole(nn.Module):
    def __init__(self, input_space, output_space) -> None:
        super(CartPole, self).__init__()
        self.fc1 = nn.Linear(input_space, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        return action
