from envs.cityflow.env import CityFlow, MaxPressure
from envs.cityflow.config_parser import id_shortcut_parased, parase_flow, parase_roadnet
import cityflow

CITYFLOW_CONFIG_ROOT_DIR = "cityflow_config/"


def make(config):
    id = id_shortcut_parased(config["id"])
    cityflow_config_dir = CITYFLOW_CONFIG_ROOT_DIR + id + "/"
    config_file_path = cityflow_config_dir + "config.json"

    eng = _build_eng(
        config_file_path=config_file_path,
        thread_num=config["thread_num"],
        save_replay=config["save_replay"],
    )

    flow_file_path = cityflow_config_dir + "flow.json"
    roadnet_file_path = cityflow_config_dir + "roadnet.json"

    flow_info = parase_flow(flow_file_path)
    intersections = parase_roadnet(roadnet_file_path, flow_info)

    max_time = flow_info["max_time"]
    interval = config["interval"]

    env = CityFlow(
        name=id,
        eng=eng,
        max_time=max_time,
        interval=interval,
        intersections=intersections,
    )
    return env


def make_mp(config):
    id = id_shortcut_parased(config["id"])
    cityflow_config_dir = CITYFLOW_CONFIG_ROOT_DIR + id + "/"
    config_file_path = cityflow_config_dir + "config.json"

    eng = _build_eng(
        config_file_path=config_file_path,
        thread_num=config["thread_num"],
        save_replay=config["save_replay"],
    )

    flow_file_path = cityflow_config_dir + "flow.json"
    roadnet_file_path = cityflow_config_dir + "roadnet.json"

    flow_info = parase_flow(flow_file_path)
    intersections = parase_roadnet(roadnet_file_path, flow_info)

    max_time = flow_info["max_time"]
    interval = config["interval"]

    env = MaxPressure(
        name=id,
        eng=eng,
        max_time=max_time,
        interval=interval,
        intersections=intersections,
    )
    return env


def _build_eng(
    config_file_path,
    thread_num,
    save_replay: bool,
):
    try:
        eng = cityflow.Engine(config_file_path, thread_num)
        eng.set_save_replay(save_replay)
    except Exception as ex:
        raise ex
    return eng
