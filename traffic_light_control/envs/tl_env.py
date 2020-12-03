from typing import Dict, List
import cityflow
import numpy as np
from basis.action import Action
from envs.intersection import Intersection
from envs.lane import Lane
from envs.phase import GraphDirection, Location
from envs.phase import Movement, Phase, TrafficStreamDirection
from basis.state import State
from envs.road import Road
from policy.static_policy import StaticPolicy


class TlEnv():
    """encapsulate cityflow by gym api
    """

    def __init__(self, config_path: str, max_time: int, interval: int,
                 thread_num=1):
        self.eng = cityflow.Engine(config_path, thread_num)

        id_core = "intersection_mid"
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
        }, eng=self.eng)
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
        }, eng=self.eng)

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
        }, eng=self.eng)

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
        }, eng=self.eng)

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
        }, eng=self.eng)

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
        }, eng=self.eng)

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
        }, eng=self.eng)

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
        }, eng=self.eng)

        roads = {}
        roads[Location.W] = {GraphDirection.OUT: west_out,
                             GraphDirection.IN: west_in}
        roads[Location.E] = {GraphDirection.OUT: east_out,
                             GraphDirection.IN: east_in}
        roads[Location.N] = {GraphDirection.OUT: north_out,
                             GraphDirection.IN: north_in}
        roads[Location.S] = {GraphDirection.OUT: south_out,
                             GraphDirection.IN: south_in}

        self.core_inter = Intersection(id_core, phase_plan, roads=roads)

        # 静态策略的路口
        self.static_inter: Dict[str, Intersection] = {}
        id_1 = "intersection_east"
        self.static_inter[id_1] = Intersection(
            id_1, [Phase([])], roads=None)
        self.static_plan: Dict[str, List[int]] = {}
        self.static_plan[id_1] = [10]
        self.static_policy = StaticPolicy()

        self.history: List[Intersection] = []
        self.time = 0
        self.max_time = max_time
        self.interval = interval

    def step(self, action: Action):
        if self.time % self.interval == 0:
            if action.keep_phase is False:
                self.core_inter.move_to_next_phase()
                self.eng.set_tl_phase(self.core_inter.id,
                                      self.core_inter.current_phase_index)

        for inter in self.static_inter.values():
            static_plan = self.static_plan[inter.id]
            keep_phase = self.static_policy.act(self.time, static_plan)
            if keep_phase is False:
                inter.move_to_next_phase()
                self.eng.set_tl_phase(inter.id, inter.current_phase_index)

        self.eng.next_step()

        state = self.__get_state()
        reward = self.__get_reward()
        done = False if self.time < self.max_time else True
        info = []
        self.time += 1

        return state, reward, done, info

    def reset(self) -> State:
        self.eng.reset()
        self.time = 0
        self.history: List[Intersection] = []
        state = State(self.core_inter)
        return state

    def set_replay_file(self, path):
        self.eng.set_replay_file(path)

    def __get_state(self) -> State:
        return State(self.core_inter)

    def __get_reward(self) -> float:
        reward = -self.__cal_total_waiting_density()
        return reward

    def __cal_total_waiting_density(self) -> float:
        intersection = self.core_inter
        total = 0.0
        for loc in Location:
            for grapDir in GraphDirection:
                for stream in TrafficStreamDirection:
                    capacity = intersection.get_road_capacity(
                        loc, grapDir, stream
                    )
                    if capacity == 0:
                        continue
                    waiting_lane = intersection.get_road_waiting_vehicles(
                        loc, grapDir, stream)
                    density = waiting_lane / capacity
                    total += density
        return total

    def __cal_total_pressure(self) -> float:
        pressure = 0.0
        for mov in Movement:
            pressure += self.__cal_pressure(mov)
        return pressure

    def __cal_pressure(self, mov: Movement):

        intersection = self.core_inter
        out_capacity = 0
        out_vehicles = 0
        in_capacity = 0
        in_vehicles = 0
        if mov == Movement.WE:
            out_capacity = intersection.get_road_capacity(
                Location.W, GraphDirection.OUT,
                TrafficStreamDirection.STRAIGHT,
            )
            out_vehicles = intersection.get_road_vehicles(
                Location.W,  GraphDirection.OUT,
                TrafficStreamDirection.STRAIGHT,
            )

            in_capacity = intersection.get_road_capacity(
                Location.E,  GraphDirection.IN,
                TrafficStreamDirection.STRAIGHT,
            )
            in_vehicles = intersection.get_road_vehicles(
                Location.E,  GraphDirection.IN,
                TrafficStreamDirection.STRAIGHT,
            )

        if mov == Movement.EW:
            out_capacity = intersection.get_road_capacity(
                Location.E, GraphDirection.OUT,
                TrafficStreamDirection.STRAIGHT
            )
            out_vehicles = intersection.get_road_vehicles(
                Location.E, GraphDirection.OUT,
                TrafficStreamDirection.STRAIGHT
            )

            in_capacity = intersection.get_road_capacity(
                Location.W, GraphDirection.IN,
                TrafficStreamDirection.STRAIGHT
            )
            in_vehicles = intersection.get_road_vehicles(
                Location.W, GraphDirection.IN,
                TrafficStreamDirection.STRAIGHT
            )
        if mov == Movement.NS:
            out_capacity = intersection.get_road_capacity(
                Location.N, GraphDirection.OUT,
                TrafficStreamDirection.STRAIGHT
            )
            out_vehicles = intersection.get_road_vehicles(
                Location.N, GraphDirection.OUT,
                TrafficStreamDirection.STRAIGHT
            )

            in_capacity = intersection.get_road_capacity(
                Location.S, GraphDirection.IN,
                TrafficStreamDirection.STRAIGHT
            )
            in_vehicles = intersection.get_road_vehicles(
                Location.S, GraphDirection.IN,
                TrafficStreamDirection.STRAIGHT
            )
        if mov == Movement.SN:
            out_capacity = intersection.get_road_capacity(
                Location.S, GraphDirection.OUT,
                TrafficStreamDirection.STRAIGHT
            )
            out_vehicles = intersection.get_road_vehicles(
                Location.S, GraphDirection.OUT,
                TrafficStreamDirection.STRAIGHT
            )

            in_capacity = intersection.get_road_capacity(
                Location.N, GraphDirection.IN,
                TrafficStreamDirection.STRAIGHT
            )
            in_vehicles = intersection.get_road_vehicles(
                Location.N, GraphDirection.IN,
                TrafficStreamDirection.STRAIGHT
            )

        out_density = out_vehicles / out_capacity
        in_density = in_vehicles / in_capacity

        pressure = abs(out_density - in_density)
        return pressure

    def __is_stuck(self) -> bool:
        return False

    def close(self):
        pass
