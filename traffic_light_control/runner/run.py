import logging
from runner.build_env import build_env
from runner.build_model import build_model
from runner.build_trainer import build_trainer, load_trainer
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
    if mode == "gym":
        # gym mode is used for test algorithm quickly
        _baseline_test(args)
        return

    env = build_env(args)
    models = build_model(args=args, env=env)

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
    logger.info("policy : {}".format(args.trainer))
    logger.info("buffer : {}".format(args.replay_buffer))

    trainer = build_trainer(args, env, models)
    trainer.save_config()

    episodes = args.episodes
    begin_time = time.time()
    _ = trainer.train(episodes)
    cost_time = (time.time() - begin_time) / 3600

    trainer.save_records()
    trainer.save_checkpoint(filename="ckpt_ending.pth")

    logger.info("total time cost {:.3f} h ".format(cost_time))
    logger.info(
        "end time is %s",
        time.asctime(time.localtime(time.time())),
    )
    logger.info("train end")
    logger.info("===== ===== =====")


def _baseline_test(args):
    trainer = hprl.test_trainer(
        trainer_type=args.trainer,
        buffer_type=args.replay_buffer,
    )
    trainer.train(args.episodes)