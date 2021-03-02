import argparse
import datetime
from hprl.util.enum import TrainnerTypes
import os
import hprl
import envs
import logging
from runner.nets import IActor, ICritic
from runner.args_paraser import create_paraser, args_validity_check

logger = logging.getLogger(__name__)

# Env Global
MAX_TIME = 3600
INTERVAL = 5

# Agent Setting
CAPACITY = 200000
LERNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
EPS_INIT = 1.0
EPS_MIN = 0.01
EPS_FRAME = 300000
UPDATE_PERIOD = 1000
INNER_EPOCH = 16
CLIP_PARAM = 0.2

# Run Setting
BASE_RECORDS_DIR = "records"
CHEKCPOINT_DIR_SUFFIX = "checkpoints"
CONFIG_DIR_SUFFIX = "configs"
LOG_DIR_SUFFIX = "log"


def run():

    paraser = create_paraser()
    args = paraser.parse_args()
    if not args_validity_check(args):
        return

    env_config = _get_env_config(args)
    env = _make_env(env_config)

    model_config = _get_model_config(
        env.get_local_state_space(),
        env.get_local_action_space(),
    )
    agents_id = env.get_agents_id()
    models = {}
    for id in agents_id:
        models[id] = _make_model(
            hprl.TrainnerTypes(args.trainer), model_config)

    mode = args.mode
    if mode == "train":
        _train(args, env, models)
    elif mode == "test":
        base_dir = f"{BASE_RECORDS_DIR}/{args.record_dir}"
        ckpt_dir = f"{base_dir}/{CHEKCPOINT_DIR_SUFFIX}"
        log_dir = f"{base_dir}/{LOG_DIR_SUFFIX}"
        trainer = hprl.load_trainer(
            env=env,
            models=models,
            checkpoint_dir=ckpt_dir,
            checkpoint_file=args.ckpt_file,
        )
        eval_episode = args.eval_episodes
        trainer.eval(eval_episode)
        trainer.log_result(log_dir)


def _train(args, env, models):

    trainer = None
    base_dir = ""
    ckpt_dir = ""
    log_dir = ""
    config_dir = ""

    if args.resume:
        base_dir = f"{BASE_RECORDS_DIR}/{args.record_dir}"
        ckpt_dir = f"{base_dir}/{CHEKCPOINT_DIR_SUFFIX}"
        log_dir = f"{base_dir}/{LOG_DIR_SUFFIX}"
        config_dir = f"{base_dir}/{CONFIG_DIR_SUFFIX}"
        trainer = hprl.load_trainer(
            env=env,
            models=models,
            checkpoint_dir=ckpt_dir,
            checkpoint_file=args.ckpt_file
        )
    else:
        base_dir, ckpt_dir, log_dir, config_dir = create_record_dir(
            BASE_RECORDS_DIR,
            args.env,
            args.trainer)

        trainer_config = _get_trainer_config(
            args=args,
            action_space=env.get_local_action_space(),
            state_space=env.get_local_state_space(),
            record_base_dir=base_dir,
            checkpoint_dir=ckpt_dir
        )
        trainer = hprl.create_trainer(
            config=trainer_config,
            env=env,
            models=models,
        )
        trainer.log_config(config_dir)
    episode = args.episodes
    eval_frequency = args.eval_frequency
    if eval_frequency <= 0:
        # if eval frequency less than zero
        # then eval after whole training process
        eval_frequency = episode + 1

    eval_episode = args.eval_episodes
    trained_time = 0
    while trained_time < episode:
        trainer_ep = min(eval_frequency, episode - trained_time)
        train_records = trainer.train(trainer_ep)
        eval_records = trainer.eval(eval_episode)
        trainer.log_result(log_dir)
        trained_time += trainer_ep
    trainer.eval(eval_episode)
    trainer.log_result(log_dir)


def _get_trainer_config(args, action_space, state_space, checkpoint_dir, record_base_dir):
    capacity = CAPACITY
    learning_rate = LERNING_RATE
    batch_size = args.batch_size
    discount_factor = DISCOUNT_FACTOR
    eps_init = EPS_INIT
    eps_min = EPS_MIN
    eps_frame = EPS_FRAME
    update_period = UPDATE_PERIOD
    action_space = action_space
    state_space = state_space
    policy_config = {
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "update_period": update_period,
        "action_space": action_space,
        "state_space": state_space,
        "eps_frame": eps_frame,
        "eps_init": eps_init,
        "eps_min": eps_min,
        "inner_epoch": INNER_EPOCH,
        "clip_param": CLIP_PARAM,
    }
    buffer_config = {
        "type": hprl.ReplayBufferTypes(args.replay_buffer),
        "capacity": capacity,
    }
    exec_config = {
        "batch_size": batch_size,
        "checkpoint_dir": checkpoint_dir,
        "record_base_dir": record_base_dir,
        "check_frequency": args.check_frequency,
    }
    trainner_config = {
        "type": hprl.TrainnerTypes(args.trainer),
        "executing": exec_config,
        "policy": policy_config,
        "buffer": buffer_config,
    }

    return trainner_config


def _get_env_config(args):
    config = {
        "id": args.env,
        "max_time": MAX_TIME,
        "interval": INTERVAL,
        "thread_num": args.env_thread_num,
        "save_replay": args.save_replay,
    }
    return config


def _get_model_config(input_space, output_space):
    config = {
        "input_space": input_space,
        "output_space": output_space,
    }
    return config


def _make_env(config):
    env = envs.make(config)
    return env


def _make_model(trainer_type, config):
    model = None
    if trainer_type == hprl.TrainnerTypes.IQL:
        acting_net = ICritic(
            input_space=config["input_space"],
            output_space=config["output_space"],
        )
        target_net = ICritic(
            input_space=config["input_space"],
            output_space=config["output_space"],
        )
        model = {
            "acting": acting_net,
            "target": target_net,
        }
    elif (trainer_type == hprl.TrainnerTypes.IAC or
          trainer_type == hprl.TrainnerTypes.PPO):

        critic_net = ICritic(
            input_space=config["input_space"],
            output_space=config["output_space"],
        )

        critic_target_net = ICritic(
            input_space=config["input_space"],
            output_space=config["output_space"],
        )

        actor_net = IActor(
            input_space=config["input_space"],
            output_space=config["output_space"],
        )

        model = {
            "critic_net": critic_net,
            "critic_target_net": critic_target_net,
            "actor_net": actor_net
        }
    else:
        raise ValueError("invalid trainer type {}".format(trainer_type))
    return model


def create_record_dir(root_dir, env_id, policy_id):
    # 创建的目录
    date = datetime.datetime.now()
    sub_dir = "{}_{}_{}_{}_{}_{}_{}_{}".format(
        env_id,
        policy_id,
        date.year,
        date.month,
        date.day,
        date.hour,
        date.minute,
        date.second,
    )
    record_dir = f"{root_dir}/{sub_dir}"
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    else:
        raise ValueError(
            "create record folder error , path exist : ", record_dir)

    checkpoint_path = f"{record_dir}/{CHEKCPOINT_DIR_SUFFIX}"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    else:
        raise ValueError(
            "create record folder error , path exist : ", checkpoint_path)

    log_path = f"{record_dir}/{LOG_DIR_SUFFIX}"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    else:
        raise ValueError(
            "create record folder error , path exist : ", log_path)

    config_path = f"{record_dir}/{CONFIG_DIR_SUFFIX}"
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    else:
        raise ValueError(
            "create record folder error , path exist : ", config_path)

    return record_dir, checkpoint_path, log_path, config_path
