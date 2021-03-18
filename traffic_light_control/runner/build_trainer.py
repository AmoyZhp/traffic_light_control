import logging
import datetime
import os
import hprl

logger = logging.getLogger(__package__)

BASE_RECORDS_DIR = "records"
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
CHEKCPOINT_DIR_SUFFIX = "checkpoints"
CONFIG_DIR_SUFFIX = "configs"
LOG_DIR_SUFFIX = "log"
ADVANTAGE_TYPE = hprl.AdvantageTypes.QMinusV


def build_trainer(args, env, models):
    if args.resume:
        trainer = load_trainer(args, env, models)
    else:
        trainer = _build_trainer(args, env, models)
    return trainer


def load_trainer(args, env, models):
    logger.info("resume record dir is {}".format(args.record_dir))
    logger.info("checkpoint file is {}".format(args.ckpt_file))
    base_dir = f"{BASE_RECORDS_DIR}/{args.record_dir}"
    ckpt_dir = ""
    if not args.ckpt_dir:
        ckpt_dir = f"{BASE_RECORDS_DIR}/{args.ckpt_dir}"
    trainer = hprl.load_trainer(
        env=env,
        models=models,
        base_dir=base_dir,
        checkpoint_dir=ckpt_dir,
        checkpoint_file=args.ckpt_file,
    )
    return trainer


def _build_trainer(args, env, models):
    recording = args.recording
    base_dir = None
    if recording:
        base_dir = _create_record_dir(
            BASE_RECORDS_DIR,
            args.env,
            args.trainer.value,
        )
        logger.info("records dir created : {}".format(base_dir))

    trainer_config = _get_trainer_config(
        args=args,
        action_space=env.get_local_action_space(),
        state_space=env.get_local_state_space(),
        record_base_dir=base_dir,
    )
    trainer = hprl.build_trainer(
        config=trainer_config,
        env=env,
        models=models,
    )
    return trainer


def _get_trainer_config(
    args,
    action_space,
    state_space,
    record_base_dir,
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
        "type": args.replay_buffer,
        "capacity": capacity,
        "alpha": args.per_alpha,
    }
    exec_config = {
        "batch_size": batch_size,
        "record_base_dir": record_base_dir,
        "ckpt_frequency": args.ckpt_frequency,
        "per_beta": args.per_beta,
        "recording": args.recording,
    }
    trainner_config = {
        "type": args.trainer,
        "executing": exec_config,
        "policy": policy_config,
        "buffer": buffer_config,
    }

    return trainner_config


def _create_record_dir(root_dir, env_id, policy_id):
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

    return record_dir