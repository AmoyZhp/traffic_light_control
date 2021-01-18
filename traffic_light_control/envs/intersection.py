import math
from pickle import INST
import numpy as np
from envs.road import Road
from typing import Dict, List
from util.enum import GraphDirection, TrafficStreamDirection


class Intersection():
    def __init__(self, id: str, phase_plan: List[List[int]],
                 roadlinks: List[Dict[GraphDirection, Road]],
                 init_phase_index: int = 0) -> None:
        self.id = id
        self.phase_plan = phase_plan
        self.current_phase_index = init_phase_index
        self.roadlinks = roadlinks
        self.roads = {}
        for rlink in self.roadlinks:
            for road in rlink.values():
                if road.id not in self.roads:
                    self.roads[road.id] = road

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
            for stream in TrafficStreamDirection:
                capacity = road.get_capacity(stream)
                if capacity == 0:
                    continue
                waiting_lane = road.get_waiting_vehicles(stream)
                density = waiting_lane / capacity
                waiting_rate += density
        return waiting_rate

    def get_pressure(self) -> float:
        pressure = 0.0
        for rlink in self.roadlinks:
            r_pressure = 0.0
            out_road = rlink[GraphDirection.OUT]
            in_road = rlink[GraphDirection.IN]
            for dir_ in TrafficStreamDirection:
                if (in_road.get_capacity(dir_) == 0
                        or out_road.get_capacity(dir_) == 0):
                    continue
                in_density = in_road.get_vehicles(
                    dir_) / in_road.get_capacity(dir_)
                out_density = out_road.get_vehicles(
                    dir_) / out_road.get_capacity(dir_)
                r_pressure += in_density - out_density

            pressure += r_pressure
        pressure = abs(pressure)
        return pressure

    def move_to_next_phase(self):
        self.current_phase_index = (
            self.current_phase_index + 1) % len(self.phase_plan)

    def to_tensor(self) -> np.ndarray:
        tensor = np.array([], dtype=np.float)
        for id_ in sorted(self.roads.keys()):
            road = self.roads[id_]
            tensor = np.hstack(
                (tensor,
                 road.to_tensor())
            )

        current_phase = self.phase_plan[
            self.current_phase_index]

        current_phase_tensor = np.zeros(12)
        for i in current_phase:
            current_phase_tensor[i] = 1

        tensor = np.hstack(
            (tensor, current_phase_tensor)
        )

        next_phase_index = (
            self.current_phase_index + 1) % len(self.phase_plan)

        next_phase = self.phase_plan[next_phase_index]
        next_phase_tensor = np.zeros(12)
        for i in next_phase:
            next_phase_tensor[i] = 1

        tensor = np.hstack(
            (tensor, next_phase_tensor)
        )
        tensor = np.expand_dims(tensor, axis=0)
        return tensor
