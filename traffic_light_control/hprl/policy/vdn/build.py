import logging
from typing import Dict

import hprl.trainer.multiagent as matrainer
import torch
import torch.nn as nn
from hprl.env import MultiAgentEnv
from hprl.policy.decorator.epsilon_greedy import MultiAgentEpsilonGreedy
from hprl.policy.policy import PolicyTypes
from hprl.policy.vdn.vdn import VDN
from hprl.recorder.recorder import Recorder
from hprl.replaybuffer import ReplayBufferTypes
from hprl.replaybuffer.common_buffer import MAgentBasisBuffer
from hprl.replaybuffer.prioritized_replay_buffer import MultiAgentPER
from hprl.replaybuffer.replay_buffer import MAgentReplayBuffer

logger = logging.getLogger(__package__)


def build_vdn_trainer(
    config: Dict,
    env: MultiAgentEnv,
    models,
    recorder: Recorder = None,
):
    buffer_config = config["buffer"]
    policy_config = config["policy"]
    executing_config = config["executing"]
    trained_iteartion = config.get("trained_iteration", 0)

    buffer_type = buffer_config["type"]
    capacity = buffer_config["capacity"]
    critic_lr = policy_config["critic_lr"]
    discount_factor = policy_config["discount_factor"]
    update_period = policy_config["update_period"]
    eps_frame = policy_config["eps_frame"]
    eps_min = policy_config["eps_min"]
    eps_init = policy_config["eps_init"]
    local_action_space = policy_config["local_action_space"]
    local_state_space = policy_config["local_state_space"]
    logger.info("=== === === build VDN trainer === === ===")

    prioritiezed = False
    agents_id = env.agents_id
    loss_fn = None
    buffer = None
    if buffer_type == ReplayBufferTypes.Prioritized:
        prioritiezed = True
        alpha = buffer_config["alpha"]
        loss_fn = _per_loss_fn
        buffer = MultiAgentPER(capacity=capacity, alpha=alpha)
        logger.info("\t buffer alpha is %s", alpha)
        logger.info("\t buffer beta is %s", executing_config["per_beta"])
    elif buffer_type == ReplayBufferTypes.Common:
        prioritiezed = False
        loss_fn = nn.MSELoss()
        buffer = MAgentBasisBuffer(capacity)
    else:
        raise ValueError("replay buffer type {} invalid".format(buffer_type))
    inner_p = VDN(
        agents_id=agents_id,
        acting_nets=models["acting_nets"],
        target_nets=models["target_nets"],
        critic_lr=critic_lr,
        discount_factor=discount_factor,
        update_period=update_period,
        local_action_space=local_action_space,
        local_state_space=local_state_space,
        loss_fn=loss_fn,
        prioritized=prioritiezed,
    )
    p = MultiAgentEpsilonGreedy(
        agents_id=agents_id,
        inner_policy=inner_p,
        eps_frame=eps_frame,
        eps_min=eps_min,
        eps_init=eps_init,
        action_space=local_action_space,
    )
    if recorder is None:
        if executing_config["recording"]:
            recorder = TorchRecorder(executing_config["record_base_dir"])
        else:
            recorder = Printer()
    trainer = matrainer.OffPolicy(
        type=PolicyTypes.VDN,
        config=executing_config,
        env=env,
        policy=p,
        buffer=buffer,
        recorder=recorder,
        trained_iter=trained_iteartion,
    )
    logger.info("=== === === build VDN trainer done === === ===")
    return trainer


def _per_loss_fn(input, target, weight):
    return torch.mean((input - target)**2 * weight)
