from typing import Dict, List
import cityflow
import numpy as np
from envs.lane import Lane
from envs.enum import Movement


class Road():
    def __init__(
        self,
        id: str,
        mov_lanes: Dict[Movement, List[Lane]],
        eng: cityflow.Engine,
    ) -> None:
        self.id = id
        self.eng = eng
        self.mov_lanes = mov_lanes
        self.id_lanes: Dict[str, Lane] = {}
        for lanes in self.mov_lanes.values():
            for lane in lanes:
                if lane.get_id() not in self.id_lanes.keys():
                    self.id_lanes[lane.get_id()] = lane

        self.stream_capacity: Dict[Movement, int] = {
            Movement.STRAIGHT: 0,
            Movement.LEFT: 0,
            Movement.RIGHT: 0,
        }
        for direction, directed_lanes in self.mov_lanes.items():
            for lane in directed_lanes:
                self.stream_capacity[direction] += lane.get_capacity()
        # each dim reprensent one movement
        self.state_space = len(self.id_lanes.keys())

    def get_capacity(self, streamDir: Movement) -> int:
        if streamDir not in self.mov_lanes.keys():
            return 0

        return self.stream_capacity[streamDir]

    def get_vehicles(self, streamDir: Movement) -> int:
        if streamDir not in self.mov_lanes.keys():
            return 0
        vehicles = 0
        directed_lanes = self.mov_lanes[streamDir]
        vehicles_dict = self.eng.get_lane_vehicle_count()
        for lane in directed_lanes:
            if lane.get_id() not in vehicles_dict.keys():
                print("key error of lane id {}".format(lane.get_id()))
            vehicles += vehicles_dict[lane.get_id()]
        return vehicles

    def get_waiting_vehicles(self, streamDir: Movement) -> int:
        if streamDir not in self.mov_lanes.keys():
            return 0
        directed_lanes = self.mov_lanes[streamDir]
        vehicles = 0
        vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        for lane in directed_lanes:
            if lane.get_id() in vehicles_dict.keys():
                vehicles += vehicles_dict[lane.get_id()]
        return vehicles

    def get_state_space(self):
        return self.state_space

    def to_tensor(self) -> np.ndarray:
        tensor = np.zeros(self.state_space)
        vehicles_dict = self.eng.get_lane_vehicle_count()
        lanes = list(self.id_lanes.values())
        for i in range(self.state_space):
            lane = lanes[i]
            vehicle_on_lane = vehicles_dict[lane.get_id()]
            density = vehicle_on_lane / lane.get_capacity()
            tensor[i] = density
        return tensor

    def __repr__(self) -> str:
        str_ = "Road[ \n"
        for dir_, lane in self.mov_lanes.items():
            str_ += " {} [{}] \n".format(dir_, lane)
        str_ += "]"
        return str_
