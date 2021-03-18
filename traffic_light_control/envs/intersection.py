import numpy as np
from envs.road import Road
from typing import Dict, List
from envs.enum import Movement, Stream

PHASE_SPACE = 12

ROAD_STATE = True


class Intersection():
    def __init__(
        self,
        id: str,
        phase_plan: List[List[int]],
        roadlinks: List[Dict[Stream, Road]],
        init_phase_index: int = 0,
    ) -> None:
        self.id = id
        self.phase_plan = phase_plan
        self.current_phase_index = init_phase_index
        self.roadlinks = roadlinks
        self.roads = {}

        self.state_space = 0
        for rlink in self.roadlinks:
            for road in rlink.values():
                if road.id not in self.roads:
                    self.roads[road.id] = road

        if ROAD_STATE:
            for road in self.roads.values():
                self.state_space += road.get_state_space()
        else:
            for rlink in self.roadlinks:
                for road in rlink.values():
                    self.state_space += road.get_state_space()
        # first phase space is current phase
        # second belong to next phase
        self.phase_space = len(self.roadlinks)
        self.state_space += 2 * self.phase_space

    def get_state_space(self):
        return self.state_space

    def get_id(self) -> str:
        return self.id

    def get_roadlinks_len(self) -> str:
        return len(self.roadlinks)

    def get_roads_ids(self):
        return list(self.roads.keys())

    def get_current_phase_index(self) -> int:
        return self.current_phase_index

    def get_current_phase(self) -> List[int]:
        return self.phase_plan[self.current_phase_index]

    def get_waiting_rate(self) -> float:
        waiting_rate = 0.0
        for road in self.roads.values():
            for stream in Movement:
                capacity = road.get_capacity(stream)
                if capacity == 0:
                    continue
                waiting_lane = road.get_waiting_vehicles(stream)
                density = waiting_lane / capacity
                waiting_rate += density
        return waiting_rate

    def get_pressure(self) -> float:
        pressure = 0.0
        cnt = 0
        for rlink in self.roadlinks:
            r_pressure = 0.0
            out_road = rlink[Stream.OUT]
            in_road = rlink[Stream.IN]
            for dir_ in Movement:
                if (in_road.get_capacity(dir_) == 0
                        or out_road.get_capacity(dir_) == 0):
                    continue
                in_density = in_road.get_vehicles(dir_) / in_road.get_capacity(
                    dir_)
                out_density = out_road.get_vehicles(
                    dir_) / out_road.get_capacity(dir_)
                traffic_mov_pres = in_density - out_density
                # scaling
                traffic_mov_pres /= 2.0
                r_pressure += traffic_mov_pres
                cnt += 1

            pressure += r_pressure
        pressure = abs(pressure)
        # scaling
        pressure /= cnt
        return pressure

    def move_to_next_phase(self):
        self.current_phase_index = (self.current_phase_index + 1) % len(
            self.phase_plan)

    def to_tensor(self) -> np.ndarray:
        if ROAD_STATE:
            tensor = self._road_state()
        else:
            tensor = self._road_link_state()

        current_phase = self.phase_plan[self.current_phase_index]

        current_phase_tensor = np.zeros(self.phase_space)
        for i in current_phase:
            current_phase_tensor[i] = 1

        tensor = np.hstack((tensor, current_phase_tensor))

        next_phase_index = (self.current_phase_index + 1) % len(
            self.phase_plan)

        next_phase = self.phase_plan[next_phase_index]
        next_phase_tensor = np.zeros(self.phase_space)
        for i in next_phase:
            next_phase_tensor[i] = 1

        tensor = np.hstack((tensor, next_phase_tensor))
        return tensor

    def _road_link_state(self):
        tensor = np.array([], dtype=np.float)
        for rlink in self.roadlinks:
            in_road = rlink[Stream.IN]
            tensor = np.hstack((tensor, in_road.to_tensor()))
            out_road = rlink[Stream.OUT]
            tensor = np.hstack((tensor, out_road.to_tensor()))
        return tensor

    def _road_state(self):
        tensor = np.array([], dtype=np.float)
        for id in sorted(self.roads.keys()):
            road = self.roads[id]
            tensor = np.hstack((tensor, road.to_tensor()))
        return tensor