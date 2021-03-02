import argparse
import datetime
import os
from trainer.independent_traffic_trainer import INTERVAL, MAX_TIME, RECORDS_ROOT_DIR
import hprl
import envs
import logging
from runner.nets import IActor, ICritic

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
INNER_EPOCH = 128
CLIP_PARAM = 0.2

# Run Setting
DEFAULT_BATCH_SIZE = 16
BASE_RECORDS_DIR = "records/"


class Runner(object):

    def run(self):

        paraser = create_paraser()
        args = paraser.parse_args()
        if not args_validity_check(args):
            paraser.print_help()
            return

        env_config = self._get_env_config(args)
        env = self._make_env(env_config)

        model_config = self._get_model_config(
            env.get_local_state_space(),
            env.get_local_action_space(),
        )
        agents_id = env.get_agents_id()
        models = {}
        for id in agents_id:
            models[id] = self._make_model(model_config)

        base_dir, ckpt_dir, log_dir, config_dir = create_record_dir(
            RECORDS_ROOT_DIR,
            args.env,
            args.trainer)

        trainer_config = self._get_trainer_config(
            args=args,
            action_space=env.get_local_action_space(),
            state_space=env.get_local_state_space(),
            checkpoint_dir=ckpt_dir
        )
        trainer = hprl.create_trainer(
            config=trainer_config,
            env=env,
            models=models,
        )

        mode = args.mode
        if mode == "train":
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

    def _get_config(self, args, extra_args):
        env_config = self._get_env_config(args)
        model_config = self._get_model_config(args, extra_args)
        trainer_config = self._get_trainer_config(args, extra_args)
        run_config = self._get_run_config(args)

        config = {
            "env": env_config,
            "model": model_config,
            "trainer": trainer_config,
            "run": run_config,
        }
        return config

    def _get_trainer_config(self, args, action_space, state_space, checkpoint_dir):
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
        }
        buffer_config = {
            "type": hprl.ReplayBufferTypes(args.replay_buffer),
            "capacity": capacity,
        }
        exec_config = {
            "batch_size": batch_size,
            "checkpoint_dir": checkpoint_dir,
            "check_frequency": args.check_frequency,
        }
        trainner_config = {
            "type": hprl.TrainnerTypes(args.trainer),
            "executing": exec_config,
            "policy": policy_config,
            "buffer": buffer_config,
        }
        return trainner_config

    def _get_env_config(self, args):
        config = {
            "id": args.env,
            "max_time": MAX_TIME,
            "interval": INTERVAL,
            "thread_num": args.env_thread_num,
            "save_replay": args.save_replay,
        }
        return config

    def _get_model_config(self, input_space, output_space):
        config = {
            "input_space": input_space,
            "output_space": output_space,
        }
        return config

    def _get_run_config(self, args, log_dir):
        eval_frequency = args.eval_frequency
        if eval_frequency <= 0:
            # if eval frequency less than zero
            # then eval after whole training process
            eval_frequency = args.episode + 1
        config = {
            "episode": args.episode,
            "eval_episode": args.eval_episode,
            "eval_frequency": eval_frequency,
            "log_dir": log_dir,
        }
        return config

    def _make_env(self, config):
        env = envs.make(config)
        return env

    def _make_model(self, config):
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
        return model


def args_validity_check(args):
    mode = args.mode
    if mode not in ["train", "test"]:
        print(" model value invalid !")
        return False
    trainer = hprl.TrainnerTypes(args.trainer)
    if trainer not in hprl.TrainnerTypes:
        print("trainer type is invalid !")
        return False
    return True


def create_paraser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, required=True,
        help="Mode of execution. Include [ train , test ]"
    )

    parser.add_argument(
        "--env", type=str, required=True,
        help="the id of the environment"
    )

    parser.add_argument(
        "--trainer", type=str, required=True,
        help="which trainer to be chosen"
    )

    parser.add_argument(
        "--episodes", type=int, default=0,
        help="episode of train times"
    )

    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="batch size of sample batch"
    )

    parser.add_argument(
        "--replay_buffer", type=str, default="Common",
        help="type of replay buffer chosen "
    )

    parser.add_argument(
        "--eval_episodes", type=int, default=1,
        help="episode of eval times"
    )

    parser.add_argument(
        "--eval_frequency", type=int, default=0,
        help="eval frequency in training process"
    )

    parser.add_argument(
        "--check_frequency", type=int, default=0,
        help="the frequency of saving checkpoint."
        "if it less than or equal zero, checkpoint would not be saved"
    )

    parser.add_argument(
        "--env_thread_num", type=int, default=1,
        help="thread number of env"
    )

    parser.add_argument(
        "save_replay", action="store_true",
        help="whether cityflow env save replay or not"
    )

    return parser


def create_record_dir(root_dir, env_id, policy_id):
    # 创建的目录
    date = datetime.datetime.now()
    sub_dir = "{}_{}_{}_{}_{}_{}_{}_{}/".format(
        env_id,
        policy_id,
        date.year,
        date.month,
        date.day,
        date.hour,
        date.minute,
        date.second,
    )
    record_dir = root_dir + sub_dir
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    else:
        raise ValueError(
            "create record folder error , path exist : ", record_dir)

    checkpoint_path = record_dir + "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    else:
        raise ValueError(
            "create record folder error , path exist : ", checkpoint_path)

    log_path = record_dir + "logs/"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    else:
        raise ValueError(
            "create record folder error , path exist : ", log_path)

    config_path = record_dir + "config/"
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    else:
        raise ValueError(
            "create record folder error , path exist : ", config_path)

    return record_dir, checkpoint_path, log_path, config_path
