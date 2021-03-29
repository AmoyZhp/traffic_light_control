from hprl.recorder.recorder import Recorder
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hprl.env import MultiAgentEnv
from hprl.policy.policy import PolicyTypes
from hprl.replaybuffer import ReplayBufferTypes
from hprl.policy.decorator import EpsilonGreedy
from hprl.replaybuffer import CommonBuffer, PrioritizedReplayBuffer
from hprl.recorder import Printer, TorchRecorder
from hprl.policy.dqn.dqn import DQN
from hprl.trainer.independent import IOffPolicyTrainer

logger = logging.getLogger(__package__)


def build_iql_trainer(
    config,
    env: MultiAgentEnv,
    models: Dict[str, nn.Module],
    recorder: Recorder = None,
    load=False,
):

    buffer_config = config["buffer"]
    policy_config = config["policy"]
    executing_config = config["executing"]

    if load:
        raw_buffer_config = buffer_config
        raw_policy_config = policy_config
        buffer_config = list(raw_buffer_config.values())[0]
        policy_config = list(raw_policy_config.values())[0]
        local_action_space = {}
        local_state_space = {}
        for id, p_config in raw_policy_config.items():
            local_action_space[id] = p_config["action_space"]
            local_state_space[id] = p_config["state_space"]
        policy_config["local_action_space"] = local_action_space
        policy_config["local_state_space"] = local_state_space
    buffer_type = buffer_config["type"]
    capacity = buffer_config["capacity"]
    critic_lr = policy_config["critic_lr"]
    discount_factor = policy_config["discount_factor"]
    update_period = policy_config["update_period"]
    eps_frame = policy_config["eps_frame"]
    eps_min = policy_config["eps_min"]
    eps_init = policy_config["eps_init"]

    logger.info("create IQL trainer")
    logger.info("\t replay buffer type : %s", buffer_type)
    logger.info("\t critic lr : %f", critic_lr)
    logger.info("\t discount factor : %f", discount_factor)
    logger.info("\t update period : %d", update_period)
    logger.info("\t eps frame : %d", eps_frame)
    logger.info("\t eps min : %f", eps_min)
    logger.info("\t eps init : %f", eps_init)

    prioritiezed = False
    agents_id = env.agents_id
    loss_fn = None
    buffers = {}
    if buffer_type == ReplayBufferTypes.Prioritized:
        prioritiezed = True
        alpha = buffer_config["alpha"]
        loss_fn = _per_loss_fn
        for id in agents_id:
            buffers[id] = PrioritizedReplayBuffer(
                capacity=capacity,
                alpha=alpha,
            )
        logger.info("\t buffer alpha is %s", alpha)
        logger.info("\t buffer beta is %s", executing_config["per_beta"])
    elif buffer_type == ReplayBufferTypes.Common:
        prioritiezed = False
        loss_fn = nn.MSELoss()
        for id in agents_id:
            buffers[id] = CommonBuffer(capacity)
    else:
        raise ValueError("replay buffer type {} invalid".format(buffer_type))

    policies = {}
    for id in agents_id:
        model = models[id]
        action_space = policy_config["local_action_space"][id]
        state_space = policy_config["local_state_space"][id]
        inner_p = DQN(
            acting_net=model["acting_net"],
            target_net=model["target_net"],
            critic_lr=critic_lr,
            discount_factor=discount_factor,
            update_period=update_period,
            action_space=action_space,
            state_space=state_space,
            prioritized=prioritiezed,
            loss_fn=loss_fn,
        )
        logger.info("\t agents %s", id)
        logger.info("\t\t action space is %s", action_space)
        logger.info("\t\t state space is %s", state_space)
        action_space = policy_config["local_action_space"][id]
        policies[id] = EpsilonGreedy(
            inner_policy=inner_p,
            eps_frame=eps_frame,
            eps_min=eps_min,
            eps_init=eps_init,
            action_space=action_space,
        )
    if recorder is None:
        if executing_config["recording"]:
            recorder = TorchRecorder(executing_config["record_base_dir"])
        else:
            recorder = Printer()
    trained_iter = config.get("trained_iteration", 0)
    trainer = IOffPolicyTrainer(
        type=PolicyTypes.IQL,
        policies=policies,
        buffers=buffers,
        env=env,
        recorder=recorder,
        config=executing_config,
        trained_iter=trained_iter,
    )
    logger.info("trainer build success")
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