from hprl.replaybuffer.prioritized_replay_buffer import MultiAgentPER
from hprl.policy.qmix.qmix import QMIX
from hprl.replaybuffer.common_buffer import MultiAgentCommonBuffer
from hprl.policy.decorator.epsilon_greedy import MultiAgentEpsilonGreedy
from hprl.recorder import Printer, TorchRecorder
from hprl.policy.policy import PolicyTypes
from hprl.replaybuffer import ReplayBufferTypes
import logging
from typing import Dict
import torch
import torch.nn as nn
from hprl.env import MultiAgentEnv
import hprl.trainer.multiagent_trainer as matrainer
from hprl.trainer import MultiAgentTraienr
logger = logging.getLogger(__package__)


def build_qmix_trainer(
    config: Dict,
    env: MultiAgentEnv,
    models,
):
    buffer_config = config["buffer"]
    policy_config = config["policy"]
    executing_config = config["executing"]

    buffer_type = buffer_config["type"]
    capacity = buffer_config["capacity"]
    critic_lr = policy_config["critic_lr"]
    discount_factor = policy_config["discount_factor"]
    update_period = policy_config["update_period"]
    eps_frame = policy_config["eps_frame"]
    eps_min = policy_config["eps_min"]
    eps_init = policy_config["eps_init"]
    action_space = policy_config["action_space"]
    state_space = policy_config["state_space"]
    logger.info("=== === === build QMIX trainer === === ===")

    prioritiezed = False
    agents_id = env.get_agents_id()
    train_fn = None
    loss_fn = None
    buffer = None
    if buffer_type == ReplayBufferTypes.Prioritized:
        prioritiezed = True
        alpha = buffer_config["alpha"]
        train_fn = matrainer.off_policy_per_train_fn
        loss_fn = _per_loss_fn
        buffer = MultiAgentPER(capacity=capacity, alpha=alpha)
        logger.info("\t buffer alpha is %s", alpha)
        logger.info("\t buffer beta is %s", executing_config["per_beta"])
    elif buffer_type == ReplayBufferTypes.Common:
        prioritiezed = False
        train_fn = matrainer.off_policy_train_fn
        loss_fn = nn.MSELoss()
        buffer = MultiAgentCommonBuffer(capacity)
    else:
        raise ValueError("replay buffer type {} invalid".format(buffer_type))
    inner_p = QMIX(
        agents_id=agents_id,
        actors_net=models["actors_net"],
        actors_target_net=models["actors_target_net"],
        critic_net=models["critic_net"],
        critic_target_net=models["critic_target_net"],
        critic_lr=critic_lr,
        discount_factor=discount_factor,
        update_period=update_period,
        action_space=action_space,
        state_space=state_space,
        loss_fn=loss_fn,
        prioritized=prioritiezed,
    )
    p = MultiAgentEpsilonGreedy(
        agents_id=agents_id,
        inner_policy=inner_p,
        eps_frame=eps_frame,
        eps_min=eps_min,
        eps_init=eps_init,
        action_space=action_space,
    )
    recorder = Printer()
    if executing_config["recording"]:
        recorder = TorchRecorder(executing_config["record_base_dir"])
        logger.info("\t training will be recorded")
    trainer = MultiAgentTraienr(
        type=PolicyTypes.QMIX,
        config=executing_config,
        env=env,
        policy=p,
        replay_buffer=buffer,
        train_fn=train_fn,
        recorder=recorder,
    )
    logger.info("=== === === build QMIX trainer done === === ===")
    return trainer


def _per_loss_fn(input, target, weight):
    return torch.mean((input - target)**2 * weight)