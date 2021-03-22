# Env Global
import envs

MAX_TIME = 3600
INTERVAL = 5


def build_env(
    env_id: str,
    thread_num: int,
    save_replay: bool,
):
    config = {
        "id": env_id,
        "max_time": MAX_TIME,
        "interval": INTERVAL,
        "thread_num": thread_num,
        "save_replay": save_replay,
    }
    env = envs.make(config)
    return env
