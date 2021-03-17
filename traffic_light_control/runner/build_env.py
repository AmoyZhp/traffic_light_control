# Env Global
import envs

MAX_TIME = 3600
INTERVAL = 5


def build_env(args):
    config = _get_env_config(args)
    env = envs.make(config)
    return env


def _get_env_config(args):
    config = {
        "id": args.env,
        "max_time": MAX_TIME,
        "interval": INTERVAL,
        "thread_num": args.env_thread_num,
        "save_replay": args.save_replay,
    }
    return config