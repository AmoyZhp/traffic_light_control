from hprl.env.multi_agent_env import MultiAgentEnv
import logging
import datetime
import os
import hprl
import hprl.recorder as hprecroder

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
    if not args.resume:
        return _build_trainer(args, env, models)
    else:
        return load_trainer(args, env, models)


def load_trainer(args, env, models):
    ckpt_file = args.ckpt_file
    recording = args.recording
    if not os.path.exists(ckpt_file):
        raise ValueError("ckeck point file not exits : {}".format(ckpt_file))
    reader = hprecroder.TorchReader()
    ckpt = reader.read_ckpt(ckpt_file)
    config = ckpt["config"]
    if not recording:
        config["executing"]["recording"] = False
    else:
        config["executing"]["recording"] = True
        record_base_dir: str = config["executing"]["record_base_dir"]
        split_str = record_base_dir.split("_")
        cnt = 0
        if split_str[-2] == "cnt":
            cnt = int(split_str[-1])
            preffix = record_base_dir.split("cnt")[0]
            new_record_base_dir = f"{preffix}cnt_{cnt+1}"
        else:
            new_record_base_dir = f"{record_base_dir}_cnt_0"
        config["executing"]["record_base_dir"] = new_record_base_dir
    logger.info("new record dir name %s", new_record_base_dir)
    trainer = hprl.load_trainer(
        env=env,
        models=models,
        ckpt=ckpt,
    )
    return trainer, trainer.recorder


def _build_trainer(args, env: MultiAgentEnv, models):
    recording = args.recording
    base_dir = None
    if recording:
        base_dir = create_record_dir(
            BASE_RECORDS_DIR,
            args.env,
            args.policy.value,
            args.record_dir_suffix,
        )
        logger.info("records dir created : {}".format(base_dir))

    if recording:
        recorder = hprecroder.TorchRecorder(base_dir)
    else:
        recorder = hprecroder.Printer()
    trainer_config = _get_trainer_config(
        args=args,
        local_state_space=env.local_state_space,
        local_action_space=env.local_action_space,
        central_state_space=env.central_state_space,
        central_action_space=env.central_action_space,
        record_base_dir=base_dir,
    )
    trainer = hprl.build_trainer(
        config=trainer_config,
        env=env,
        models=models,
        recorder=recorder,
    )
    recorder.write_config(config=env.setting, filename="env_setting.json")
    recorder.write_config(config=repr(models), filename="model_setting.data")
    recorder.write_config(
        config=trainer.get_config(),
        filename="initial_config.json",
    )
    return trainer, recorder


def _get_trainer_config(
    args,
    local_state_space,
    local_action_space,
    central_state_space,
    central_action_space,
    record_base_dir,
):
    capacity = args.capacity
    critic_lr = args.critic_lr
    actor_lr = args.actor_lr
    batch_size = args.batch_size
    if batch_size <= 0:
        raise ValueError(
            "Please input valid batch size, there is not default value")
    discount_factor = args.gamma
    eps_init = EPS_INIT
    eps_min = EPS_MIN
    eps_frame = args.eps_frame
    update_period = args.update_period
    inner_epoch = args.inner_epoch
    clip_param = args.clip_param
    policy_config = {
        "critic_lr": critic_lr,
        "actor_lr": actor_lr,
        "discount_factor": discount_factor,
        "update_period": update_period,
        "central_state_space": central_state_space,
        "central_action_space": central_action_space,
        "local_action_space": local_action_space,
        "local_state_space": local_state_space,
        "eps_frame": eps_frame,
        "eps_init": eps_init,
        "eps_min": eps_min,
        "inner_epoch": inner_epoch,
        "clip_param": clip_param,
        "advg_type": args.advg_type,
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
        "type": args.policy,
        "executing": exec_config,
        "policy": policy_config,
        "buffer": buffer_config,
    }

    return trainner_config


def create_record_dir(
    root_dir,
    env_id,
    policy_id,
    suffix="",
):
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
    if suffix:
        record_dir = f"{record_dir}_{suffix}"
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    else:
        raise ValueError("create record folder error , path exist : ",
                         record_dir)

    return record_dir