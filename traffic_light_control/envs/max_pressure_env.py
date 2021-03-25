from envs.enum import Movement, Stream
from envs.road import Road
import logging
from hprl.util.typing import Action
from envs.intersection import Intersection
from typing import Dict, List, Tuple
import hprl
import numpy as np

logger = logging.getLogger(__name__)


class MaxPressureEnv():
    def __init__(
        self,
        eng,
        name: str,
        max_time: int,
        interval: int,
        intersections: Dict[str, Intersection],
    ):
        self.eng = eng
        self.name = name
        self.intersections = intersections
        self.intersections_id = sorted(list(self.intersections.keys()))
        self.interval = interval
        self.max_time = max_time
        self.time = 0

    def step(self, action: hprl.Action):
        self._process_action(action)
        for _ in range(self.interval):
            self.eng.next_step()

        self.time += self.interval
        next_state = self._compute_state()
        reward = self._compute_reward()
        done = False if self.time < self.max_time else True
        info = self._get_info()
        return next_state, reward, done, info

    def _get_info(self):
        info = {"avg_travel_time": self.eng.get_average_travel_time()}
        return info

    def _process_action(self, action: Action):
        for id in self.intersections_id:
            act = action.local[id]
            self.eng.set_tl_phase(id, act)

    def _compute_state(self) -> hprl.State:
        local_state = {}
        for id in self.intersections_id:
            pressure = 0.0
            item = self.intersections[id]
            phase_plans = item.phase_plan
            roadlinks = item.roadlinks
            phases_pressure = []
            for i in range(len(phase_plans)):
                for road_link_index in phase_plans[i]:
                    rlink = roadlinks[road_link_index]
                    pressure += abs(self.cal_roadlink_pressure(rlink))
                pressure = pressure
                phases_pressure.append(pressure)
            local_state[id] = phases_pressure

        state = hprl.State(central={}, local=local_state)
        return state

    def cal_roadlink_pressure(self, rlink):
        r_pressure = 0.0
        out_road = rlink[Stream.OUT]
        in_road = rlink[Stream.IN]
        for dir_ in Movement:
            if (in_road.get_capacity(dir_) == 0
                    or out_road.get_capacity(dir_) == 0):
                continue
            in_density = in_road.get_vehicles(dir_) / in_road.get_capacity(
                dir_)
            out_density = out_road.get_vehicles(dir_) / out_road.get_capacity(
                dir_)
            traffic_mov_pres = in_density - out_density
            r_pressure += traffic_mov_pres
        pressure = r_pressure
        return pressure

    def _compute_reward(self):
        central_reward = 0.0
        for id in self.intersections_id:
            intersection = self.intersections[id]
            r = -intersection.get_pressure()
            central_reward += r
        central_reward /= len(self.intersections)
        return central_reward

    def reset(self) -> np.ndarray:
        self.eng.reset()
        self.time = 0
        return self._compute_state()

    def get_central_action_space(self):
        return self.central_action_space

    def get_local_action_space(self):
        return self.local_action_space

    def get_central_state_space(self):
        return self.central_state_space

    def get_local_state_space(self):
        return self.local_state_space

    def get_agents_id(self) -> List[str]:
        return self.intersections_id

    def get_env_name(self):
        return self.name

    @property
    def setting(self):
        _setting = {
            "max_time": self.max_time,
            "interval": self.interval,
            "env_name": self.get_env_name(),
            "agents_id": self.get_agents_id(),
            "central_state_space": self.get_central_state_space(),
            "central_action_space": self.get_central_action_space(),
            "local_action_space": self.get_local_action_space(),
            "local_state_space": self.get_local_state_space(),
        }
        return _setting
