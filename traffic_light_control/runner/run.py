import datetime
import json
import logging
import os
import time
from typing import List

import envs
import hprl
import hprl.recorder as hprecorder
from hprl.util.typing import Action

from runner.args_paraser import args_validity_check, create_paraser
from runner.build_trainer import BASE_RECORDS_DIR
from runner.util import log_record, plot_avg_travel_time

logger = logging.getLogger(__name__)
INTERVAL = 5

ROOT_OUTPUT_DIR = "records"


def run():
    paraser = create_paraser()
    args = paraser.parse_args()
    if not args_validity_check(args):
        return

    mode = args.mode

    resume = args.resume
    recording = args.recording if mode != "eval" else False
    output_dir = ""
    if mode == "train":
        if resume:
            ckpt_path = args.ckpt_path
            config = init_override_config(args)
            trainer = hprl.load_trainer(
                ckpt_path=ckpt_path,
                override_config=config,
                recording=recording,
            )
            output_dir = trainer.output_dir
        else:
            config = init_trainer_config(args)
            if recording:
                output_dir = create_output_dir(
                    root_dir=ROOT_OUTPUT_DIR,
                    env_id=args.env,
                    policy_id=args.policy.value,
                    suffix=args.record_dir_suffix,
                )
                hprl.log_to_file(output_dir)
            config["trainer"]["output_dir"] = output_dir
            trainer = hprl.trainer.build_iql_trainer(config)
        episodes = args.episodes
        ckpt_frequency = args.ckpt_frequency
        _train(
            trainer=trainer,
            episodes=episodes,
            recording=recording,
            ckpt_frequency=ckpt_frequency,
            output_dir=output_dir,
        )
    elif mode == "eval":
        ckpt_path = args.ckpt_path
        override_config = {}
        override_config["env"] = {
            "thread_num": args.env_thread_num,
            "save_replay": args.save_replay,
        }
        trainer = hprl.load_trainer(
            ckpt_path=ckpt_path,
            override_config=override_config,
            recording=recording,
        )
        _eval(trainer=trainer, episodes=args.episodes, output_dir="records")


def _train(
    trainer: hprl.Trainer,
    episodes: int,
    recording: bool,
    ckpt_frequency: int,
    output_dir: str,
):
    logger.info("===== ===== =====")
    logger.info("train beigin")
    logger.info(
        "begin time : %s",
        time.asctime(time.localtime(time.time())),
    )

    if not recording:
        ckpt_frequency = 0
    begin_time = time.time()
    records = trainer.train(
        episodes=episodes,
        ckpt_frequency=ckpt_frequency,
        log_record_fn=log_record,
    )
    cost_time = (time.time() - begin_time) / 3600

    if recording:
        trainer.save_checkpoint()
        records = trainer.get_records()

        hprl.recorder.plot_summation_rewards(
            records=records,
            fig_path=output_dir,
            save_fig=True,
        )
        hprl.recorder.plot_avg_rewards(
            records=records,
            fig_path=output_dir,
            save_fig=True,
        )
        plot_avg_travel_time(
            records=records,
            fig_path=output_dir,
            save_fig=True,
        )
    logger.info("total time cost {:.3f} h ".format(cost_time))
    logger.info(
        "end time is %s",
        time.asctime(time.localtime(time.time())),
    )
    logger.info("train end")
    logger.info("===== ===== =====")


def _eval(
    trainer: hprl.Trainer,
    episodes: int,
    output_dir="",
):
    records = trainer.eval(episodes=episodes, log_record_fn=log_record)
    if output_dir:
        hprl.recorder.plot_summation_rewards(
            records=records,
            fig_path=output_dir,
            save_fig=True,
        )
        hprl.recorder.plot_avg_rewards(
            records=records,
            fig_path=output_dir,
            save_fig=True,
        )
        plot_avg_travel_time(
            records=records,
            fig_path=output_dir,
            save_fig=True,
        )
        sum_rewards = []
        avg_rewards = []
        avg_travel_times = []
        for record in records:
            rewards = []
            # for one episode
            for r in record.rewards:
                rewards.append(r.central)
            sum_reward = sum(rewards)
            avg_reward = sum_reward / len(rewards)
            avg_travel_time = record.infos[-1]["avg_travel_time"]
            sum_rewards.append(sum_reward)
            avg_rewards.append(avg_reward)
            avg_travel_times.append(avg_travel_time)

        avg_sum_reward = sum(sum_rewards) / len(sum_rewards)
        avg_avg_reward = sum(avg_rewards) / len(avg_rewards)
        avg_avg_travel_time = sum(avg_travel_times) / len(avg_travel_times)
        result = {
            "sum_reward": avg_sum_reward,
            "avg_rewards": avg_avg_reward,
            "avg_travel_time": avg_avg_travel_time,
        }
        result_path = f"{output_dir}/eval_result.json"
        with open(result_path, "w") as f:
            json.dump(result, f)


def init_trainer_config(args):
    capacity = args.capacity
    critic_lr = args.critic_lr
    actor_lr = args.actor_lr
    batch_size = args.batch_size
    if batch_size <= 0:
        raise ValueError(
            "Please input valid batch size, there is not default value")
    discount_factor = args.gamma
    eps_init = args.eps_init
    eps_min = args.eps_min
    eps_frame = args.eps_frame
    update_period = args.update_period
    inner_epoch = args.inner_epoch
    clip_param = args.clip_param

    policy_config = {
        "type": args.policy,
        "model_id": args.policy.value,
        "critic_lr": critic_lr,
        "actor_lr": actor_lr,
        "discount_factor": discount_factor,
        "update_period": update_period,
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
    env_config = {
        "type": "CityFlow",
        "id": args.env,
        "interval": INTERVAL,
        "thread_num": args.env_thread_num,
        "save_replay": args.save_replay,
    }
    trainer_config = {
        "type": args.policy,
        "batch_size": batch_size,
        "per_beta": args.per_beta,
        "recording": args.recording,
    }
    trainner_config = {
        "trainer": trainer_config,
        "policy": policy_config,
        "buffer": buffer_config,
        "env": env_config,
    }

    return trainner_config


def init_override_config(args):
    override_config = {}
    override_config["env"] = {
        "thread_num": args.env_thread_num,
        "save_replay": args.save_replay,
    }
    return override_config


def create_output_dir(
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
    output_dir = f"{root_dir}/{sub_dir}"
    if suffix:
        output_dir = f"{output_dir}_{suffix}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        raise ValueError("create record folder error , path exist : ",
                         output_dir)

    return output_dir


def _baseline_test(args):
    episodes = 0
    if args.env == "gym":
        trainer = hprl.gym_baseline_trainer(
            trainer_type=args.policy,
            buffer_type=args.replay_buffer,
            batch_size=args.batch_size,
        )
    elif args.policy == hprl.PolicyTypes.MP:
        max_pressure_eval(args)
        return
    else:
        trainer, episodes = build_baseline_trainer(
            env_id=args.env,
            policy_type=args.policy,
            buffer_type=args.replay_buffer,
            batch_size=args.batch_size,
        )
    if args.episodes > 0:
        episodes = args.episodes
    trainer.train(episodes)


def max_pressure_eval(args):
    config = {
        "id": args.env,
        "interval": 5,
        "thread_num": args.env_thread_num,
        "save_replay": args.save_replay,
    }
    env = envs.make_mp_cityflow(config)
    state = env.reset()
    sum_reward = 0.0
    avg_travel_time = 0
    while True:
        actions = {}
        for id, local_s in state.local.items():
            action = 0
            max_pressure = local_s[0]
            for i in range(len(local_s)):
                if local_s[i] > max_pressure:
                    action = i
                    max_pressure = local_s[i]
            actions[id] = action
        state, reward, done, info = env.step(Action(local=actions))
        sum_reward += reward.central
        if done.central:
            avg_travel_time = info["avg_travel_time"]
            break
    result = {
        "reward": sum_reward,
        "avg_travel_time": avg_travel_time,
    }
    recording = args.recording
    base_dir = None
    if recording:
        base_dir = create_record_dir(
            BASE_RECORDS_DIR,
            args.env,
            args.policy.value,
        )
        logger.info("records dir created : {}".format(base_dir))
        filepath = f"{base_dir}/result.json"
        with open(filepath, "w") as f:
            json.dump(result, f)
