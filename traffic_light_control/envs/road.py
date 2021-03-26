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

    def get_capacity(self, stream_dir: Movement) -> int:
        if stream_dir not in self.mov_lanes.keys():
            return 0

        return self.stream_capacity[stream_dir]

    def get_vehicles(self, stream_dir: Movement) -> int:
        if stream_dir not in self.mov_lanes.keys():
            return 0
        vehicles = 0
        directed_lanes = self.mov_lanes[stream_dir]
        vehicles_dict = self.eng.get_lane_vehicle_count()
        for lane in directed_lanes:
            if lane.get_id() not in vehicles_dict.keys():
                print("key error of lane id {}".format(lane.get_id()))
            vehicles += vehicles_dict[lane.get_id()]
        return vehicles

    def get_density(self) -> int:
        cnt = 0
        density = 0.0
        for dir in Movement:
            if not self.get_capacity(dir):
                continue
            density += self.get_vehicles(dir) / self.get_capacity(dir)
            cnt += 1
        density /= cnt
        return density

    def get_waiting_vehicles(self, stream_dir: Movement) -> int:
        if stream_dir not in self.mov_lanes.keys():
            return 0
        directed_lanes = self.mov_lanes[stream_dir]
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
