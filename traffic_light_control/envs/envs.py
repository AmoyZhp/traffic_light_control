
from typing import Dict, List
from envs.independent_traffic_env import IndependentTrafficEnv
from envs.intersection import Intersection
from envs.lane import Lane
from envs.phase import Phase
from envs.road import Road
import cityflow
import json
from util.enum import *


def make(config):
    id_ = config["id"]
    if id_ == "multi_agent_independent":
        return __get_multi_agent_independent(config)
    elif id_ == "single_complete_1x1" or id_ == "single_complete_1x1_static":
        return __get_single_complete_1x1_test(config)
    elif id_ == "single_agent_simplest":
        return __get_single_agent_simplest(config)
    else:
        print("invalid environment id {}".format(id_))


def __get_multi_agent_independent(config):
    pass


def __get_single_complete_1x1(config):
    cityflow_config_dir = config["cityflow_config_dir"]
    cityflow_config_file = cityflow_config_dir + "config.json"
    eng = cityflow.Engine(cityflow_config_file, config["thread_num"])
    eng.set_save_replay(config["save_replay"])
    # 应该跟 config 中的 light phase 一致
    # 先按照 config 里先手工写好 Phase
    phase_plan: List[Phase] = []
    phase_plan.append(Phase(movements=[
        Movement.WS, Movement.EN, Movement.NW, Movement.SE,
        Movement.WE, Movement.EW, ]))
    phase_plan.append(Phase(movements=[
        Movement.WS, Movement.EN, Movement.NW, Movement.SE,
        Movement.WN, Movement.ES, ]))
    phase_plan.append(Phase(movements=[
        Movement.WS, Movement.EN, Movement.NW, Movement.SE,
        Movement.SN, Movement.NS]))
    phase_plan.append(Phase(movements=[
        Movement.WS, Movement.EN, Movement.NW, Movement.SE,
        Movement.SW, Movement.NE]))
    west_out = Road(id="road_west_to_mid_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_west_to_mid_1_0", 15),
        ],
        TrafficStreamDirection.RIGHT: [
            Lane("road_west_to_mid_1_1", 15),
        ],
        TrafficStreamDirection.LEFT: [
            Lane("road_west_to_mid_1_2", 15),
        ]
    }, eng=eng)
    west_in = Road(id="road_mid_to_west_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_mid_to_west_1_0", 15),
        ],
        TrafficStreamDirection.RIGHT: [
            Lane("road_mid_to_west_1_1", 15),
        ],
        TrafficStreamDirection.LEFT: [
            Lane("road_mid_to_west_1_2", 15),
        ]
    }, eng=eng)

    east_in = Road(id="road_mid_to_east_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_mid_to_east_1_0", 15)
        ],
        TrafficStreamDirection.RIGHT: [
            Lane("road_mid_to_east_1_1", 15)
        ],
        TrafficStreamDirection.LEFT: [
            Lane("road_mid_to_east_1_2", 15)
        ]
    }, eng=eng)

    east_out = Road(id="road_east_to_mid_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_east_to_mid_1_0", 15),
        ],
        TrafficStreamDirection.RIGHT: [
            Lane("road_east_to_mid_1_1", 15),
        ],
        TrafficStreamDirection.LEFT: [
            Lane("road_east_to_mid_1_2", 15),
        ]
    }, eng=eng)

    north_out = Road(id="road_north_to_mid_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_north_to_mid_1_0", 15)
        ],
        TrafficStreamDirection.RIGHT: [
            Lane("road_north_to_mid_1_1", 15)
        ],
        TrafficStreamDirection.LEFT: [
            Lane("road_north_to_mid_1_2", 15)
        ]
    }, eng=eng)

    north_in = Road(id="road_mid_to_north_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_mid_to_north_1_0", 15)
        ],
        TrafficStreamDirection.RIGHT: [
            Lane("road_mid_to_north_1_1", 15)
        ],
        TrafficStreamDirection.LEFT: [
            Lane("road_mid_to_north_1_2", 15)
        ]
    }, eng=eng)

    south_in = Road(id="road_mid_to_south_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_mid_to_south_1_0", 15)
        ],
        TrafficStreamDirection.RIGHT: [
            Lane("road_mid_to_south_1_1", 15)
        ],
        TrafficStreamDirection.LEFT: [
            Lane("road_mid_to_south_1_2", 15)
        ]
    }, eng=eng)

    south_out = Road(id="road_south_to_mid_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_south_to_mid_1_0", 15)
        ],
        TrafficStreamDirection.RIGHT: [
            Lane("road_south_to_mid_1_1", 15)
        ],
        TrafficStreamDirection.LEFT: [
            Lane("road_south_to_mid_1_2", 15)
        ]
    }, eng=eng)

    roads = {}
    roads[Location.W] = {GraphDirection.OUT: west_out,
                         GraphDirection.IN: west_in}
    roads[Location.E] = {GraphDirection.OUT: east_out,
                         GraphDirection.IN: east_in}
    roads[Location.N] = {GraphDirection.OUT: north_out,
                         GraphDirection.IN: north_in}
    roads[Location.S] = {GraphDirection.OUT: south_out,
                         GraphDirection.IN: south_in}

    intersections = {}
    id_ = "intersection_mid"
    intersections["intersection_mid"] = Intersection(
        id_, phase_plan=phase_plan, roads=roads)

    env = IndependentTrafficEnv(
        id_=config["id"],
        eng=eng, max_time=config["max_time"],
        interval=config["interval"], intersections=intersections)
    return env


def __get_single_agent_simplest(config):
    cityflow_config_path = "config/config_single_simple.json"
    eng = cityflow.Engine(cityflow_config_path, config["thread_num"])
    eng.set_save_replay(config["save_replay"])

    phase_plan: List[Phase] = []
    phase_plan.append(Phase(movements=[Movement.WE, Movement.EW]))
    phase_plan.append(Phase(movements=[Movement.SN, Movement.NS]))
    west_out = Road(id="road_west_to_mid_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_west_to_mid_1_0", 15),
        ]
    }, eng=eng)
    west_in = Road(id="road_mid_to_west_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_mid_to_west_1_0", 15),
        ]
    }, eng=eng)

    east_in = Road(id="road_mid_to_east_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_mid_to_east_1_0", 15)
        ]
    }, eng=eng)
    east_out = Road(id="road_east_to_mid_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_east_to_mid_1_0", 15),
        ]
    }, eng=eng)

    north_out = Road(id="road_north_to_mid_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_north_to_mid_1_0", 15)
        ]
    }, eng=eng)

    north_in = Road(id="road_mid_to_north_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_mid_to_north_1_0", 15)
        ]
    }, eng=eng)

    south_in = Road(id="road_mid_to_south_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_mid_to_south_1_0", 15)
        ]
    }, eng=eng)

    south_out = Road(id="road_south_to_mid_1", lanes={
        TrafficStreamDirection.STRAIGHT: [
            Lane("road_south_to_mid_1_0", 15)
        ]
    }, eng=eng)

    roads = {}
    roads[Location.W] = {GraphDirection.OUT: west_out,
                         GraphDirection.IN: west_in}
    roads[Location.E] = {GraphDirection.OUT: east_out,
                         GraphDirection.IN: east_in}
    roads[Location.N] = {GraphDirection.OUT: north_out,
                         GraphDirection.IN: north_in}
    roads[Location.S] = {GraphDirection.OUT: south_out,
                         GraphDirection.IN: south_in}

    intersections = {}
    id_ = "intersection_mid"
    intersections["intersection_mid"] = Intersection(
        id_, phase_plan=phase_plan, roads=roads)

    env = IndependentTrafficEnv(
        id_=config["id"],
        eng=eng, max_time=config["max_time"],
        interval=config["interval"], intersections=intersections)
    return env


def __get_single_complete_1x1_test(config):
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
