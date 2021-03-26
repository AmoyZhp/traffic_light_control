from envs.enum import Movement, Stream
from envs.road import Road
import logging
from hprl.util.typing import Action
from envs.intersection import Intersection, RoadLink
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
        self._eng_step()
        self.time += self.interval
        next_state = self._compute_state()
        reward = self._compute_reward()
        done = False if self.time < self.max_time else True
        info = self._get_info()
        return next_state, reward, done, info

    def _eng_step(self):
        for _ in range(self.interval):
            self.eng.next_step()
        self._update()

    def _update(self):
        vehicles_dict = self.eng.get_lane_vehicle_count()
        waiting_vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        for inter in self.intersections.values():
            inter.update(
                vehicles_data_pool=vehicles_dict,
                waiting_vehicles_data_pool=waiting_vehicles_dict,
            )

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
            item = self.intersections[id]
            phase_plans: List[List[int]] = item.phase_plan
            roadlinks: List[RoadLink] = item.roadlinks
            phases_pressure = []
            for phase in phase_plans:
                pressure = 0.0
                for rlink_index in phase:
                    rlink = roadlinks[rlink_index]
                    pressure += rlink.pressure
                pressure = abs(pressure)
                phases_pressure.append(pressure)
            local_state[id] = phases_pressure

        state = hprl.State(central={}, local=local_state)
        return state

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
        self._update()
        return self._compute_state()

    def get_env_name(self):
        return self.name
