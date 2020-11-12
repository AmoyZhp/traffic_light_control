from typing import Dict, List, Set
import cityflow
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

    def __init__(self, config_path: str, thread_num=1):
        self.eng = cityflow.Engine(config_path, thread_num)

        id_core = "intersection_mid"
        # 应该跟 config 中的 light phase 一致
        # 先按照 config 里先手工写好 Phase
        phase_plan: List[Phase] = []
        phase_plan.append(Phase(movements=[Movement.WE, Movement.EW]))
        phase_plan.append(Phase(movements=[Movement.SN, Movement.NS]))
        west_out = Road(id="road_west_to_mid_1", lanes={
            TrafficStreamDirection.STRAIGHT: [
                Lane("road_west_to_mid_1_0", 15),
                Lane("road_west_to_mid_1_1", 15),
                Lane("road_west_to_mid_1_2", 15)
            ]
        }, eng=self.eng)
        west_in = Road(id="road_mid_to_west_1", lanes={
            TrafficStreamDirection.STRAIGHT: [
                Lane("road_mid_to_west_1_0", 15),
            ]
        }, eng=self.eng)

        east_in = Road(id="road_mid_to_east_1", lanes={
            TrafficStreamDirection.STRAIGHT: [
                Lane("road_mid_to_east_1_0", 15)
            ]
        }, eng=self.eng)
        east_out = Road(id="road_east_to_mid_1", lanes={
            TrafficStreamDirection.STRAIGHT: [
                Lane("road_east_to_mid_1_0", 15),
            ]
        }, eng=self.eng)

        north_out = Road(id="road_north_to_mid_1", lanes={
            TrafficStreamDirection.STRAIGHT: [
                Lane("road_north_to_mid_1_0", 15)
            ]
        }, eng=self.eng)

        north_in = Road(id="road_mid_to_north_1", lanes={
            TrafficStreamDirection.STRAIGHT: [
                Lane("road_mid_to_north_1_0", 15)
            ]
        }, eng=self.eng)

        south_in = Road(id="road_mid_to_south_1", lanes={
            TrafficStreamDirection.STRAIGHT: [
                Lane("road_mid_to_south_1_0", 15)
            ]
        }, eng=self.eng)

        south_out = Road(id="road_south_to_mid_1", lanes={
            TrafficStreamDirection.STRAIGHT: [
                Lane("road_south_to_mid_1_0", 15)
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
            id_1, [Phase([]), Phase([])], roads=None)
        self.static_plan: Dict[str, List[int]] = {}
        self.static_plan[id_1] = [10, 40]
        self.static_policy = StaticPolicy()

        self.history: List[Intersection] = []
        self.time = 0

    def step(self, action: Action):
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
        self.time += 1

        state = 0.0
        reward = self.__get_reward()
        done = False
        info = []

        return state, reward, done, info

    def reset(self) -> State:
        self.eng.reset()
        state = State({}, None, None)
        return state

    def __get_state(self) -> State:
        return State({}, None, None)

    def __get_reward(self) -> float:
        pressure = self.__cal_pressure()
        reward = -pressure
        return reward

    def __cal_pressure(self) -> float:
        intersection = self.core_inter
        current_phase = intersection.get_current_phase()
        pressure = 0.0
        for mov in current_phase.get_movements():
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
            # TO DO : 其他方向的判断
            move_pressure = abs(out_density - in_density)
            pressure += move_pressure

        return pressure

    def __is_stuck(self) -> bool:
        return False

    def close(self):
        pass
