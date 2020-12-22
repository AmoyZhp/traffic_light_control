
from typing import List
from envs.independent_traffic_env import IndependentTrafficEnv
from envs.intersection import Intersection
from envs.lane import Lane
from envs.phase import GraphDirection, Location
from envs.phase import Movement, Phase, TrafficStreamDirection
from envs.road import Road
import cityflow


def make(id_: str, config):
    if id_ == "multi_agent_independent":
        return __get_multi_agent_independent(config)
    elif id_ == "single_agent_complete":
        return __get_single_agent_complete(config)
    elif id_ == "single_agent_simplest":
        return __get_single_agent_simplest(config)
    else:
        print("invalid environment id {}".format(id_))


def __get_multi_agent_independent(config):
    pass


def __get_single_agent_complete(config):
    cityflow_config_path = "config/config.json"
    eng = cityflow.Engine(cityflow_config_path, config["thread_num"])
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
        eng=eng, max_time=config["max_time"],
        interval=config["interval"], intersections=intersections)
    return env


def __get_single_agent_simplest(config):
    cityflow_config_path = "config/config_single_simple.json"
    eng = cityflow.Engine(cityflow_config_path, config["thread_num"])
    print(config["save_replay"])
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
        eng=eng, max_time=config["max_time"],
        interval=config["interval"], intersections=intersections)
    return env
