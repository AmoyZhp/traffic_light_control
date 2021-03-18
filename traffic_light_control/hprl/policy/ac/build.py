from hprl.policy.ac.ppo import PPO
import hprl.trainer.independent_learner_trainer as ilearner
from hprl.trainer.independent_learner_trainer import IndependentLearnerTrainer
from hprl.recorder.torch_recorder import TorchRecorder
from hprl.recorder.printer import Printer
from hprl.util.enum import AdvantageTypes, TrainnerTypes
from hprl.policy.ac.actor_critic import ActorCritic
import logging
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from hprl.env import MultiAgentEnv

logger = logging.getLogger(__package__)


def build_ppo_trainer(
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
    advg_type = policy_config["advg_type"]

    logger.info("create IAC trainer")
    logger.info("\t critic lr : %f", critic_lr)
    logger.info("\t discount factor : %f", discount_factor)
    logger.info("\t update period : %d", update_period)
    logger.info("\t inner epoch : %d", inner_epoch)
    logger.info("\t clip param : %f : ", clip_param)

    agents_id = env.get_agents_id()
    train_fn = ilearner.on_policy_train_fn
    loss_fn = nn.MSELoss()
    policies = {}
    for id in agents_id:
        model = models[id]
        critic_net = model["critic_net"]
        critic_target_net = model["critic_target_net"]
        actor_net = model["actor_net"]
        action_space = policy_config["action_space"][id]
        state_space = policy_config["state_space"][id]
        policies[id] = PPO(
            critic_net=critic_net,
            critic_target_net=critic_target_net,
            inner_epoch=inner_epoch,
            clip_param=clip_param,
            actor_net=actor_net,
            critic_lr=critic_lr,
            actor_lr=actor_lr,
            discount_factor=discount_factor,
            update_period=update_period,
            action_space=action_space,
            state_space=state_space,
            critic_loss_fn=loss_fn,
            advantage_type=advg_type,
        )

    recorder = Printer()
    if executing_config["recording"]:
        recorder = TorchRecorder(executing_config["record_base_dir"])
        logger.info("\t training will be recorded")
    trainer = IndependentLearnerTrainer(
        type=TrainnerTypes.PPO,
        policies=policies,
        env=env,
        train_fn=train_fn,
        recorder=recorder,
        config=executing_config,
        replay_buffers=None,
    )
    return trainer


def build_ac_trainer(
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
    advg_type = policy_config["advg_type"]

    logger.info("create IAC trainer")
    logger.info("\t critic lr : %f", critic_lr)
    logger.info("\t discount factor : %f", discount_factor)
    logger.info("\t update period : %d", update_period)

    agents_id = env.get_agents_id()
    train_fn = ilearner.on_policy_train_fn
    loss_fn = nn.MSELoss()
    policies = {}
    for id in agents_id:
        model = models[id]
        critic_net = model["critic_net"]
        critic_target_net = model["critic_target_net"]
        actor_net = model["actor_net"]
        action_space = policy_config["action_space"][id]
        state_space = policy_config["state_space"][id]
        policies[id] = ActorCritic(
            critic_net=critic_net,
            critic_target_net=critic_target_net,
            actor_net=actor_net,
            critic_lr=critic_lr,
            actor_lr=actor_lr,
            discount_factor=discount_factor,
            update_period=update_period,
            action_space=action_space,
            state_space=state_space,
            critic_loss_fn=loss_fn,
            advantage_type=advg_type,
        )

    recorder = Printer()
    if executing_config["recording"]:
        recorder = TorchRecorder(executing_config["record_base_dir"])
        logger.info("\t training will be recorded")
    trainer = IndependentLearnerTrainer(
        type=TrainnerTypes.IAC,
        policies=policies,
        env=env,
        train_fn=train_fn,
        recorder=recorder,
        config=executing_config,
        replay_buffers=None,
    )
    return trainer


def get_ppo_test_setting():
    config, model = get_ac_test_setting()
    config["type"] = TrainnerTypes.PPO
    config["policy"]["inner_epoch"] = 16
    config["policy"]["clip_param"] = 0.2
    return config, model


def get_ac_test_setting():
    critic_lr = 1e-3
    actor_lr = 1e-3
    batch_size = 16
    discount_factor = 0.99
    update_period = 100
    action_space = 2
    state_space = 4
    advg_type = AdvantageTypes.RewardToGO
    policy_config = {
        "critic_lr": critic_lr,
        "actor_lr": actor_lr,
        "discount_factor": discount_factor,
        "update_period": update_period,
        "advg_type": advg_type,
        "action_space": {},
        "state_space": {},
    }
    buffer_config = {}
    exec_config = {
        "batch_size": batch_size,
        "recording": True,
        "ckpt_frequency": 0,
        "record_base_dir": "records/gym_test",
    }
    trainner_config = {
        "type": TrainnerTypes.IAC,
        "executing": exec_config,
        "policy": policy_config,
        "buffer": buffer_config,
    }
    actor_net = CartPolePG(
        input_space=state_space,
        output_space=action_space,
    )

    critic_net = CartPole(
        input_space=state_space,
        output_space=action_space,
    )

    critic_target_net = CartPole(
        input_space=state_space,
        output_space=action_space,
    )

    model = {
        "actor_net": actor_net,
        "critic_net": critic_net,
        "critic_target_net": critic_target_net,
    }

    return trainner_config, model


class CartPolePG(nn.Module):
    def __init__(self, input_space, output_space) -> None:
        super(CartPolePG, self).__init__()
        self.fc1 = nn.Linear(input_space, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, output_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = F.softmax(self.fc3(x), dim=-1)
        return action


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