from hprl.replaybuffer.common_buffer import MultiAgentCommonBuffer
from hprl.policy.decorator.epsilon_greedy import MultiAgentEpsilonGreedy
from hprl.policy.vdn.vdn import VDN
from hprl.recorder import Printer, TorchRecorder
from hprl.replaybuffer.replay_buffer import MultiAgentReplayBuffer
from hprl.util.enum import ReplayBufferTypes, TrainnerTypes
import logging
from typing import Dict
import torch
import torch.nn as nn
from hprl.env import MultiAgentEnv
import hprl.trainer.multiagent_trainer as matrainer
from hprl.trainer import MultiAgentTraienr
logger = logging.getLogger(__package__)


def build_vdn_trainer(
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
    logger.info("=== === === build VDN trainer === === ===")

    prioritiezed = False
    agents_id = env.get_agents_id()
    train_fn = None
    loss_fn = None
    buffer = None
    if buffer_type == ReplayBufferTypes.Prioritized:
        raise NotImplementedError
    elif buffer_type == ReplayBufferTypes.Common:
        prioritiezed = False
        train_fn = matrainer.off_policy_train_fn
        loss_fn = nn.MSELoss()
        buffer = MultiAgentCommonBuffer(capacity)
    else:
        raise ValueError("replay buffer type {} invalid".format(buffer_type))
    inner_p = VDN(
        agents_id=agents_id,
        acting_nets=models["acting_nets"],
        target_nets=models["target_nets"],
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
        type=TrainnerTypes.VDN,
        config=executing_config,
        env=env,
        policy=p,
        replay_buffer=buffer,
        train_fn=train_fn,
        recorder=recorder,
    )
    logger.info("=== === === build VDN trainer done === === ===")
    return trainer