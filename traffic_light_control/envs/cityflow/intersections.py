from typing import Dict, List

import numpy as np
from envs.cityflow.road import Road, RoadLink

Phase = List[int]


class Intersection():
    def __init__(
        self,
        id: str,
        phase_plan: List[Phase],
        roadlinks: List[RoadLink],
        roads: Dict[str, Road],
        init_phase_index: int = 0,
    ) -> None:
        self._id = id
        self._phases_plan = phase_plan
        self._current_phase_index = init_phase_index
        self._roadlinks = roadlinks
        self._roads = roads
        self._validity()

        self._state_sapce = 0
        for road in self._roads.values():
            self._state_sapce += road.state_space
        # first phase space is current phase
        # second belong to next phase
        self._phase_space = len(self._roadlinks)
        self._state_sapce += self._phase_space

    def update(
        self,
        vehicles_data_pool,
        waiting_vehicles_data_pool,
        vehicles_speed_pool,
    ):
        for road in self._roads.values():
            road.update(
                vehicles_data_pool=vehicles_data_pool,
                waiting_vehicles_data_pool=waiting_vehicles_data_pool,
                vehicles_speed_pool=vehicles_speed_pool,
            )

    def move_to_next_phase(self):
        self._current_phase_index = (self._current_phase_index + 1) % len(
            self._phases_plan)

    @property
    def avg_spped_rate(self) -> float:
        avg_speed_sum = 0.0
        vehicles_count = 0
        for road in self._roads.values():
            avg_speed_sum += road.avg_speed_sum
            vehicles_count += road.vehicles
        if not vehicles_count:
            return 0.0
        avg_speed_rate = avg_speed_sum / vehicles_count

        return avg_speed_rate

    @property
    def waiting_rate(self) -> float:
        waiting_rate = 0.0
        for road in self._roads.values():
            waiting_rate = road.waiting_vehicles / road.vehicles
        return waiting_rate

    @property
    def pressure(self) -> float:
        pressure = 0.0
        for rlink in self._roadlinks:
            pressure += rlink.pressure
        pressure = abs(pressure)
        # scaling
        pressure /= len(self._roadlinks)
        return pressure

    @property
    def tensor(self) -> np.ndarray:
        tensor = np.array([], dtype=np.float)
        for id in sorted(self._roads.keys()):
            road = self._roads[id]
            tensor = np.hstack((tensor, road.tensor))
        current_phase = self._phases_plan[self._current_phase_index]

        current_phase_tensor = np.zeros(self._phase_space)
        for i in current_phase:
            current_phase_tensor[i] = 1

        tensor = np.hstack((tensor, current_phase_tensor))
        return tensor

    @property
    def roads(self):
        return self._roads

    @property
    def state_space(self):
        return self._state_sapce

    @property
    def id(self) -> str:
        return self._id

    @property
    def roadlinks_count(self) -> str:
        return len(self._roadlinks)

    @property
    def roads_id(self):
        return list(self._roads.keys())

    @property
    def phase_index(self) -> int:
        return self._current_phase_index

    @phase_index.setter
    def phase_index(self, index):
        assert index >= 0 and index < len(self._phases_plan)
        self._current_phase_index = index

    @property
    def current_phase(self) -> List[int]:
        return self._phases_plan[self._current_phase_index]

    def _validity(self):
        for rlink in self._roadlinks:
            start_road = rlink.start_road
            if start_road.id not in self._roads:
                raise ValueError(
                    "road %s and road link mistatched ",
                    start_road.id,
                )
            end_road = rlink.end_road
            if end_road.id not in self._roads:
                raise ValueError(
                    "road %s and road link mistatched ",
                    end_road.id,
                )
