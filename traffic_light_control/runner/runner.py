import datetime
import os
import logging
import time

import hprl
import envs
from runner.nets import IActor, ICritic, COMACritic
from runner.args_paraser import create_paraser, args_validity_check

logger = logging.getLogger(__name__)

# Env Global
MAX_TIME = 3600
INTERVAL = 5

# Agent Setting
CAPACITY = 2048
LERNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
EPS_INIT = 1.0
EPS_MIN = 0.01
EPS_FRAME = 300000
UPDATE_PERIOD = 1000
INNER_EPOCH = 32
CLIP_PARAM = 0.2

# Run Setting
BASE_RECORDS_DIR = "records"
CHEKCPOINT_DIR_SUFFIX = "checkpoints"
CONFIG_DIR_SUFFIX = "configs"
LOG_DIR_SUFFIX = "log"
ADVANTAGE_TYPE = hprl.AdvantageTypes.QMinusV


def run():

    paraser = create_paraser()
    args = paraser.parse_args()
    if not args_validity_check(args):
        return

    env_config = _get_env_config(args)
    env = _make_env(env_config)

    agents_id = env.get_agents_id()
    model_config = {
        "central_state": env.get_central_state_space(),
        "local_state": env.get_local_state_space(),
        "central_action": env.get_central_action_space(),
        "local_action": env.get_local_action_space(),
    }
    models = _make_model(
        hprl.TrainnerTypes(args.trainer),
        model_config,
        agents_id,
    )

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
        trainer.save_records(log_dir)


def _train(args, env, models):
    logger.info("===== ===== =====")
    logger.info("train process beigin")
    logger.info(" env is {}".format(args.env))
    logger.info(" policy is {}".format(args.trainer))

    trainer = None
    base_dir = ""
    ckpt_dir = ""
    log_dir = ""
    config_dir = ""

    if args.resume:
        logger.info("resume record dir is {}".format(args.record_dir))
        logger.info("checkpoint file is {}".format(args.ckpt_file))

        base_dir = f"{BASE_RECORDS_DIR}/{args.record_dir}"
        ckpt_dir = f"{base_dir}/{CHEKCPOINT_DIR_SUFFIX}"
        log_dir = f"{base_dir}/{LOG_DIR_SUFFIX}"
        config_dir = f"{base_dir}/{CONFIG_DIR_SUFFIX}"
        trainer = hprl.load_trainer(env=env,
                                    models=models,
                                    checkpoint_dir=ckpt_dir,
                                    checkpoint_file=args.ckpt_file)
    else:
        base_dir, ckpt_dir, log_dir, config_dir = create_record_dir(
            BASE_RECORDS_DIR, args.env, args.trainer)
        logger.info("records dir created : {}".format(base_dir))

        trainer_config = _get_trainer_config(
            args=args,
            action_space=env.get_local_action_space(),
            state_space=env.get_local_state_space(),
            record_base_dir=base_dir,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir)
        trainer = hprl.build_trainer(
            config=trainer_config,
            env=env,
            models=models,
        )
        trainer.save_config(config_dir)
    episode = args.episodes
    eval_frequency = args.eval_frequency
    if eval_frequency <= 0:
        # if eval frequency less than zero
        # then eval after whole training process
        eval_frequency = episode + 1

    eval_episode = args.eval_episodes
    trained_time = 0
    begin_time = time.time()

    while trained_time < episode:
        trainer_ep = min(eval_frequency, episode - trained_time)
        train_records = trainer.train(trainer_ep)
        eval_records = trainer.eval(eval_episode)
        trainer.save_records(log_dir)
        trained_time += trainer_ep
    trainer.eval(eval_episode)
    trainer.save_records(log_dir)
    trainer.save_checkpoint(ckpt_dir, "ckpt_ending.pth")
    cost_time = (time.time() - begin_time) / 3600
    logger.info("total time cost {:.3f} h ".format(cost_time))
    logger.info("train end")
    logger.info("===== ===== =====")


def _get_trainer_config(
    args,
    action_space,
    state_space,
    checkpoint_dir,
    record_base_dir,
    log_dir,
):
    capacity = args.capacity
    critic_lr = args.critic_lr
    actor_lr = args.actor_lr
    batch_size = args.batch_size
    discount_factor = args.gamma
    eps_init = EPS_INIT
    eps_min = EPS_MIN
    eps_frame = args.eps_frame
    update_period = args.update_period
    inner_epoch = args.inner_epoch
    clip_param = args.clip_param
    action_space = action_space
    state_space = state_space
    policy_config = {
        "critic_lr": critic_lr,
        "actor_lr": actor_lr,
        "discount_factor": discount_factor,
        "update_period": update_period,
        "action_space": action_space,
        "state_space": state_space,
        "eps_frame": eps_frame,
        "eps_init": eps_init,
        "eps_min": eps_min,
        "inner_epoch": inner_epoch,
        "clip_param": clip_param,
        "advantage_type": ADVANTAGE_TYPE,
    }
    buffer_config = {
        "type": hprl.ReplayBufferTypes(args.replay_buffer),
        "capacity": capacity,
        "alpha": args.per_alpha,
    }
    exec_config = {
        "batch_size": batch_size,
        "checkpoint_dir": checkpoint_dir,
        "record_base_dir": record_base_dir,
        "log_dir": log_dir,
        "ckpt_frequency": args.ckpt_frequency,
        "per_beta": args.per_beta,
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


def _make_env(config):
    env = envs.make(config)
    return env


def _make_model(trainer_type, config, agents_id):
    if trainer_type == hprl.TrainnerTypes.IQL:
        return _make_iql_model(config, agents_id)
    elif trainer_type == hprl.TrainnerTypes.IQL_PS:
        return _make_iql_ps_model(config, agents_id)
    elif (trainer_type == hprl.TrainnerTypes.IAC
          or trainer_type == hprl.TrainnerTypes.PPO):
        return _make_ac_model(config, agents_id)
    elif (trainer_type == hprl.TrainnerTypes.PPO_PS):
        return _make_ppo_ps_model(config, agents_id)
    elif trainer_type == hprl.TrainnerTypes.VDN:
        return _make_vdn_model(config, agents_id)
    elif trainer_type == hprl.TrainnerTypes.COMA:
        return _make_coma_model(config, agents_id)
    elif trainer_type == hprl.TrainnerTypes.IQL_PER:
        return _make_iql_model(config, agents_id)
    else:
        raise ValueError("invalid trainer type {}".format(trainer_type))


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
        raise ValueError("create record folder error , path exist : ",
                         record_dir)

    checkpoint_path = f"{record_dir}/{CHEKCPOINT_DIR_SUFFIX}"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    else:
        raise ValueError("create record folder error , path exist : ",
                         checkpoint_path)

    log_path = f"{record_dir}/{LOG_DIR_SUFFIX}"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    else:
        raise ValueError("create record folder error , path exist : ",
                         log_path)

    config_path = f"{record_dir}/{CONFIG_DIR_SUFFIX}"
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    else:
        raise ValueError("create record folder error , path exist : ",
                         config_path)

    return record_dir, checkpoint_path, log_path, config_path


def _make_iql_model(config, agents_id):
    models = {}
    print(f"iql model config {config}")
    for id in agents_id:
        acting_net = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
        target_net = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
        models[id] = {
            "acting_net": acting_net,
            "target_net": target_net,
        }
    return models


def _make_iql_ps_model(config, agents_id):
    models = {}
    acting_net = ICritic(
        input_space=config["local_state"][id],
        output_space=config["local_action"][id],
    )
    target_net = ICritic(
        input_space=config["local_state"][id],
        output_space=config["local_action"][id],
    )
    for id in agents_id:
        models[id] = {
            "acting_net": acting_net,
            "target_net": target_net,
        }
    return models


def _make_ac_model(config, agents_id):
    models = {}
    for id in agents_id:
        critic_net = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )

        critic_target_net = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )

        actor_net = IActor(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )

        models[id] = {
            "critic_net": critic_net,
            "critic_target_net": critic_target_net,
            "actor_net": actor_net
        }
    return models


def _make_ppo_ps_model(config, agents_id):
    models = {}
    critic_net = ICritic(
        input_space=config["local_state"][id],
        output_space=config["local_action"][id],
    )

    critic_target_net = ICritic(
        input_space=config["local_state"][id],
        output_space=config["local_action"][id],
    )

    actor_net = IActor(
        input_space=config["local_state"][id],
        output_space=config["local_action"][id],
    )
    for id in agents_id:
        models[id] = {
            "critic_net": critic_net,
            "critic_target_net": critic_target_net,
            "actor_net": actor_net
        }
    return models


def _make_vdn_model(config, agents_id):
    acting_nets = {}
    target_nets = {}
    for id in agents_id:
        acting_nets[id] = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
        target_nets[id] = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
    model = {
        "acting_nets": acting_nets,
        "target_nets": target_nets,
    }
    return model


def _make_coma_model(config, agents_id):
    critic_input_space = (config["central_state"] + config["local_state"][id] +
                          len(agents_id) +
                          len(agents_id) * config["local_action"][id])
    critic_net = COMACritic(critic_input_space, config["local_action"][id])
    target_critic_net = COMACritic(critic_input_space,
                                   config["local_action"][id])
    actors_net = {}
    for id in agents_id:
        actors_net[id] = IActor(config["local_state"][id],
                                config["local_action"][id])
    model = {
        "critic_net": critic_net,
        "target_critic_net": target_critic_net,
        "actors_net": actors_net,
    }
    return model