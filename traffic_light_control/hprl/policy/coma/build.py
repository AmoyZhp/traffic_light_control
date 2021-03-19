from hprl.replaybuffer.common_buffer import MultiAgentCommonBuffer
from hprl.policy.coma.coma import COMA
from hprl.recorder.torch_recorder import TorchRecorder
from hprl.recorder.printer import Printer
from hprl.policy.policy import AdvantageTypes, PolicyTypes
from hprl.trainer.multiagent_trainer import MultiAgentTraienr
import hprl.trainer.multiagent_trainer as matrainer
import logging
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from hprl.env import MultiAgentEnv

logger = logging.getLogger(__package__)


def build_coma_trainer(
    config,
    env: MultiAgentEnv,
    models: Dict[str, nn.Module],
):
    policy_config = config["policy"]
    executing_config = config["executing"]

    critic_lr = policy_config["critic_lr"]
    actor_lr = policy_config["actor_lr"]
    discount_factor = policy_config["discount_factor"]
    update_period = policy_config["update_period"]
    inner_epoch = policy_config["inner_epoch"]
    clip_param = policy_config["clip_param"]

    logger.info("create coma trainer")
    logger.info("\t critic lr : %f", critic_lr)
    logger.info("\t discount factor : %f", discount_factor)
    logger.info("\t update period : %d", update_period)
    logger.info("\t inner epoch : %d", inner_epoch)
    logger.info("\t clip param : %f : ", clip_param)

    agents_id = env.get_agents_id()
    train_fn = matrainer.on_policy_train_fn
    loss_fn = nn.MSELoss()
    critic_net = models["critic_net"]
    critic_target_net = models["critic_target_net"]
    actors_net = models["actors_net"]
    action_space = policy_config["action_space"]
    state_space = policy_config["state_space"]
    policy = COMA(
        agents_id=agents_id,
        critic_net=critic_net,
        critic_target_net=critic_target_net,
        actors_net=actors_net,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        discount_factor=discount_factor,
        update_period=update_period,
        local_action_space=action_space,
        local_state_space=state_space,
        clip_param=clip_param,
        inner_epoch=inner_epoch,
    )
    recorder = Printer()
    if executing_config["recording"]:
        recorder = TorchRecorder(executing_config["record_base_dir"])
        logger.info("\t training will be recorded")
    trainer = MultiAgentTraienr(
        type=PolicyTypes.COMA,
        env=env,
        config=executing_config,
        policy=policy,
        replay_buffer=MultiAgentCommonBuffer(0),
        train_fn=train_fn,
        recorder=recorder,
    )
    return trainer