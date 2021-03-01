from envs.traffic_light_ctrl_env import TrafficLightCtrlEnv
from typing import Dict, List
from envs.intersection import Intersection
from envs.lane import Lane
from envs.road import Road
import cityflow
import json
import math
from envs.enum import Stream, Movement

CITYFLOW_CONFIG_ROOT_DIR = "cityflow_config/"


def get_default_config_for_single():
    config = {
        "id": "1x1",
        "thread_num": 1,
        "save_replay": False,
        "max_time": 3600,
        "interval": 5,
    }
    return config


def get_default_config_for_multi():
    config = {
        "id": "2x2",
        "thread_num": 1,
        "save_replay": False,
        "max_time": 3600,
        "interval": 5,
    }
    return config


def make(config):
    return _get_env_by_roadnet(config)


def _id_shortcut_parased(id_):
    if id_ == "1x3":
        id_ = "syn_1x3_gaussian_500_1h"
    elif id_ == "1x1":
        id_ = "hangzhou_1x1_bc-tyc_18041607_1h"
    elif id_ == "2x2":
        id_ = "syn_2x2_gaussian_500_1h"
    # 都不匹配则直接返回
    return id_


def _get_env_by_roadnet(config):
    id_ = _id_shortcut_parased(config["id"])
    cityflow_config_dir = CITYFLOW_CONFIG_ROOT_DIR + id_ + "/"
    cityflow_config_file = cityflow_config_dir + "config.json"

    eng = cityflow.Engine(cityflow_config_file, config["thread_num"])
    eng.set_save_replay(config["save_replay"])

    roadnet_file = cityflow_config_dir + "roadnet.json"
    flow_file = cityflow_config_dir + "flow.json"

    intersections = _parse_cityflow_file(
        roadnet_file=roadnet_file,
        flow_file=flow_file, config_file=cityflow_config_dir, eng=eng)

    env = TrafficLightCtrlEnv(
        id_=config["id"],
        eng=eng, max_time=config["max_time"],
        interval=config["interval"], intersections=intersections)
    return env


def _parse_cityflow_file(roadnet_file,
                         flow_file,
                         config_file, eng) -> Dict[str, Intersection]:

    flow_json = json.load(open(flow_file))
    flow_info = _parase_flow_info(flow_json)

    roadnet_json = json.load(open(roadnet_file))
    roads_info = _parase_roads_info(roadnet_json["roads"])
    for r in roads_info.values():
        r["capacity"] = int(r["length"] / flow_info["vehicle"]["proportion"])

    inters = {}
    for inter_json in roadnet_json["intersections"]:
        if inter_json['virtual']:
            continue

        roadlinks = _parase_roadlink(inter_json["roadLinks"], roads_info, eng)

        phase_plan = _parase_phase_plan(
            inter_json["trafficLight"]["lightphases"])

        inter_id = inter_json["id"]

        inters[inter_id] = Intersection(
            id=inter_id, roadlinks=roadlinks, phase_plan=phase_plan)
    return inters


def _parase_flow_info(flow_json):
    flow = {}
    vehicle_json = flow_json[0]["vehicle"]
    vehicle_proportion = float(
        vehicle_json["length"]) + float(vehicle_json["minGap"])
    flow["vehicle"] = {
        "proportion": vehicle_proportion
    }

    return flow


def _parase_roads_info(roads_json):
    roads_info = {}
    for r_json in roads_json:
        id_ = r_json["id"]
        points = r_json["points"]
        x1 = float(points[0]["x"])
        y1 = float(points[0]["y"])
        x2 = float(points[1]["x"])
        y2 = float(points[1]["y"])
        length = math.sqrt(
            math.pow((x1-x2), 2) + math.pow((y1-y2), 2)
        )

        roads_info[id_] = {
            "length": length
        }

    return roads_info


def _parase_roadlink(roadlinks_json, roads_info, eng):

    # 以下两个字典作为中间变量，用来得到最后 roadlinks
    # roadlinks temp 记录每条 roadlink 的 road id
    roadlinks_temp: List[Dict[Stream, str]] = []
    # roads_lanes_temp 用来记录每条 road 下的 lanes
    roads_lanes_temp = {}

    for roadlink_json in roadlinks_json:

        out_road_id = roadlink_json["startRoad"]
        in_road_id = roadlink_json["endRoad"]
        roadlinks_temp.append({
            Stream.OUT: out_road_id,
            Stream.IN: in_road_id
        })

        link_type = Movement(roadlink_json["type"])
        lanelinks = roadlink_json["laneLinks"]
        out_lanes = []
        in_lanes = []
        for lane_json in lanelinks:
            start_index = lane_json["startLaneIndex"]
            end_index = lane_json["endLaneIndex"]
            out_lane_id = "{}_{}".format(out_road_id, start_index)
            in_lane_id = "{}_{}".format(in_road_id, end_index)
            out_lanes.append(
                Lane(out_lane_id, roads_info[out_road_id]["capacity"]))
            in_lanes.append(
                Lane(in_lane_id, roads_info[in_road_id]["capacity"]))

        if out_road_id not in roads_lanes_temp.keys():
            roads_lanes_temp[out_road_id] = {}
        if in_road_id not in roads_lanes_temp.keys():
            roads_lanes_temp[in_road_id] = {}
        roads_lanes_temp[out_road_id][link_type] = out_lanes
        roads_lanes_temp[in_road_id][link_type] = in_lanes

    roadlinks = []
    for temp in roadlinks_temp:
        out_id = temp[Stream.OUT]
        in_id = temp[Stream.IN]
        out_lanes = roads_lanes_temp[out_id]
        in_lanes = roads_lanes_temp[in_id]
        rlink = {
            Stream.OUT: Road(
                id=out_id,
                lanes=out_lanes,
                eng=eng
            ),
            Stream.IN: Road(
                id=in_id,
                lanes=in_lanes,
                eng=eng
            )
        }
        roadlinks.append(rlink)
    return roadlinks


def _parase_phase_plan(phase_plan_json):
    phase_plan = []
    for phase_json in phase_plan_json:
        phase_plan.append(phase_json["availableRoadLinks"])
    return phase_plan
