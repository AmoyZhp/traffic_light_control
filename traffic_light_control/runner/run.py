from hprl.util.typing import Action
import json
import logging
from typing import List
from runner.build_env import build_env
from runner.build_model import build_model
from runner.build_trainer import BASE_RECORDS_DIR, build_trainer, load_trainer, create_record_dir
from runner.build_baseline_trainer import build_baseline_trainer
import time
import hprl
import envs
from runner.args_paraser import create_paraser, args_validity_check

logger = logging.getLogger(__package__)


def run():

    paraser = create_paraser()
    args = paraser.parse_args()
    if not args_validity_check(args):
        return

    mode = args.mode
    if mode == "baseline":
        # gym mode is used for test algorithm quickly
        _baseline_test(args)
        return

    env = build_env(
        env_id=args.env,
        thread_num=args.env_thread_num,
        save_replay=args.save_replay,
    )
    models = build_model(policy=args.policy, env=env)

    if mode == "train":
        _train(args, env, models)
    elif mode == "test":
        _eval(args, env, models)


def _eval(args, env, models):
    trainer = load_trainer(args, env, models)
    episodes = args.episodes
    trainer.eval(episodes)
    trainer.save_records()


def _train(args, env, models):
    logger.info("===== ===== =====")
    logger.info("train beigin")
    logger.info(
        "begin time : %s",
        time.asctime(time.localtime(time.time())),
    )
    logger.info("env : {}".format(args.env))
    logger.info("policy : {}".format(args.policy))
    logger.info("buffer : {}".format(args.replay_buffer))

    trainer, recorder = build_trainer(args, env, models)

    episodes = args.episodes
    begin_time = time.time()
    _ = trainer.train(episodes)
    cost_time = (time.time() - begin_time) / 3600

    trainer.save_records()
    trainer.save_checkpoint(filename="ckpt_ending.pth")
    exp_record = {"cost_time": cost_time, "episodes": episodes}
    recorder.write_config(config=exp_record, filename="exp_record.json")

    logger.info("total time cost {:.3f} h ".format(cost_time))
    logger.info(
        "end time is %s",
        time.asctime(time.localtime(time.time())),
    )
    logger.info("train end")
    logger.info("===== ===== =====")


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
    env = envs.make_mp_env(config)
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
            actions[id] = action

        state, reward, done, info = env.step(Action(local=actions))
        sum_reward += reward
        if done:
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