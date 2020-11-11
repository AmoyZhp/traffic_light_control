from typing import Dict, List, Set
import cityflow
from basis.action import Action
from envs.intersection import Intersection
from envs.phase import Direction, Movement, Phase
from basis.state import State
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
        self.core_inter = Intersection(id_core, phase_plan)

        # 静态策略的路口
        self.static_inter: Dict[str, Intersection] = {}
        id_1 = "intersection_east"
        self.static_inter[id_1] = Intersection(id_1, [Phase([]), Phase([])])
        self.static_plan: Dict[str, List[int]] = {}
        self.static_plan[id_1] = [10, 40]
        self.static_policy = StaticPolicy()

        self.history: List[State] = []
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

        state = None
        reward = 0.0
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
        intersection_id = "intersection_mid"
        pressure = self.__cal_pressure(intersection_id)
        reward = -pressure
        # 如果发生完全的阻塞则给一个非常大的负数奖励
        if self.__is_stuck(intersection_id):
            reward = reward - 1000000

    def __cal_pressure(self, intersection_id: str) -> float:
        intersection = self.intersections[intersection_id]
        current_phase = intersection.get_current_phase()
        pressure = 0.0
        for mov in current_phase.get_movements():
            out_density = 0.0
            in_density = 0.0
            if mov == Movement.WE:
                out_capacity = intersection.get_roads_capacity(
                    Movement.WE, Direction.W)
                out_vehicles = intersection.get_roads_vehicles(
                    Movement.WE, Direction.W)
                out_density = out_vehicles / out_capacity

                in_capacity = intersection.get_roads_capacity(
                    Movement.WE, Direction.E)
                in_vehicles = intersection.get_roads_vehicles(
                    Movement.WE, Direction.E)
                in_density = in_vehicles / in_capacity
            # TO DO : 其他方向的判断
            move_pressure = abs(out_density - in_density)
            pressure += move_pressure

        return pressure

    def __is_stuck(self, intersection_id: str) -> bool:
        intersection = self.intersections[intersection_id]
        current_phase = intersection.get_current_phase()
        for mov in current_phase.get_movements():
            if mov == Movement.NS:
                out_waiting_vehicles = intersection.get_road_waiting_vehicles(
                    Movement.NS, Direction.N)
                last_state = self.history[self.time-1]
                # 不相等的话说明刚刚变更了路灯状态，无法判断是否阻塞
                if last_state.current_phase.equal(current_phase) is False:
                    return False
        return False

    def close(self):
        pass
