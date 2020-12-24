
from typing import Dict, List
from envs.independent_traffic_env import IndependentTrafficEnv
from envs.intersection import Intersection
from envs.lane import Lane
from envs.road import Road
import cityflow
import json
from util.enum import *


def make(config):
    id_ = config["id"]
    if id_ == "multi_agent_independent":
        return __get_multi_agent_independent(config)
    elif id_ == "single_complete_1x1" or id_ == "single_complete_1x1_static":
        return __get_single_complete_1x1(config)
    else:
        print("invalid environment id {}".format(id_))


def __get_multi_agent_independent(config):
    pass


def __get_single_complete_1x1(config):
    cityflow_config_dir = config["cityflow_config_dir"]
    cityflow_config_file = cityflow_config_dir + "config.json"

    eng = cityflow.Engine(cityflow_config_file, config["thread_num"])
    eng.set_save_replay(config["save_replay"])

    intersections = __parse_roadnet(cityflow_config_dir + "roadnet.json", eng)
    env = IndependentTrafficEnv(
        id_=config["id"],
        eng=eng, max_time=config["max_time"],
        interval=config["interval"], intersections=intersections)
    return env


def __parse_roadnet(roadnet_file, eng) -> Dict[str, Intersection]:
    roadnet = json.load(open(roadnet_file))
    inters = {}
    for inter_json in roadnet["intersections"]:
        if inter_json['virtual']:
            continue

        roadlinks = __parase_roadlink(inter_json["roadLinks"], eng)

        phase_plan = __parase_phase_plan(
            inter_json["trafficLight"]["lightphases"])

        inter_id = inter_json["id"]

        inters[inter_id] = Intersection(
            id=inter_id, roadlinks=roadlinks, phase_plan=phase_plan)
    return inters


def __parase_roadlink(roadlinks_json, eng):

    # 以下两个字典作为中间变量，用来得到最后 roadlinks
    # roadlinks temp 记录每条 roadlink 的 road id
    roadlinks_temp: List[Dict[GraphDirection, str]] = []
    # roads_lanes_temp 用来记录每条 road 下的 lanes
    roads_lanes_temp = {}

    for roadlink_json in roadlinks_json:

        out_road_id = roadlink_json["startRoad"]
        in_road_id = roadlink_json["endRoad"]
        roadlinks_temp.append({
            GraphDirection.OUT: out_road_id,
            GraphDirection.IN: in_road_id
        })

        link_type = TrafficStreamDirection(roadlink_json["type"])
        lanelinks = roadlink_json["laneLinks"]
        out_lanes = []
        in_lanes = []
        for lane_json in lanelinks:
            start_index = lane_json["startLaneIndex"]
            end_index = lane_json["endLaneIndex"]
            out_lane_id = "{}_{}".format(out_road_id, start_index)
            in_lane_id = "{}_{}".format(in_road_id, end_index)
            out_lanes.append(Lane(out_lane_id, 15))
            in_lanes.append(Lane(in_lane_id, 15))

        if out_road_id not in roads_lanes_temp.keys():
            roads_lanes_temp[out_road_id] = {}
        if in_road_id not in roads_lanes_temp.keys():
            roads_lanes_temp[in_road_id] = {}
        roads_lanes_temp[out_road_id][link_type] = out_lanes
        roads_lanes_temp[in_road_id][link_type] = in_lanes

    roadlinks = []
    for temp in roadlinks_temp:
        out_id = temp[GraphDirection.OUT]
        in_id = temp[GraphDirection.IN]
        out_lanes = roads_lanes_temp[out_id]
        in_lanes = roads_lanes_temp[in_id]
        rlink = {
            GraphDirection.OUT: Road(
                id=out_id,
                lanes=out_lanes,
                eng=eng
            ),
            GraphDirection.IN: Road(
                id=in_id,
                lanes=in_lanes,
                eng=eng
            )
        }
        roadlinks.append(rlink)
    return roadlinks


def __parase_phase_plan(phase_plan_json):
    phase_plan = []
    for phase_json in phase_plan_json:
        phase_plan.append(phase_json["availableRoadLinks"])
    return phase_plan
