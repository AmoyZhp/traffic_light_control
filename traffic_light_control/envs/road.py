from typing import Dict, List
import cityflow
import numpy as np
from envs.lane import Lane
from envs.enum import Movement


class Road():
    def __init__(
        self,
        id: str,
        lanes: Dict[Movement, List[Lane]],
        eng: cityflow.Engine,
    ) -> None:
        self.id = id
        self.eng = eng
        self.lanes = lanes
        self.stream_capacity: Dict[Movement, int] = {
            Movement.STRAIGHT: 0,
            Movement.LEFT: 0,
            Movement.RIGHT: 0,
        }
        for direction, directed_lanes in self.lanes.items():
            for lane in directed_lanes:
                self.stream_capacity[direction] += lane.get_capacity()
        # each dim reprensent one movement
        self.state_space = len(Movement)

    def get_capacity(self, stream_dir: Movement) -> int:
        if stream_dir not in self.lanes.keys():
            return 0

        return self.stream_capacity[stream_dir]

    def get_vehicles(self, stream_dir: Movement) -> int:
        if stream_dir not in self.lanes.keys():
            return 0
        vehicles = 0
        directed_lanes = self.lanes[stream_dir]
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
        if stream_dir not in self.lanes.keys():
            return 0
        directed_lanes = self.lanes[stream_dir]
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
        dire = Movement.LEFT
        tensor[0] = (0 if self.get_capacity(dire) == 0 else
                     self.get_vehicles(dire) / self.get_capacity(dire))

        dire = Movement.STRAIGHT
        tensor[1] = (0 if self.get_capacity(dire) == 0 else
                     self.get_vehicles(dire) / self.get_capacity(dire))

        dire = Movement.RIGHT
        tensor[2] = (0 if self.get_capacity(dire) == 0 else
                     self.get_vehicles(dire) / self.get_capacity(dire))
        return tensor

    def __repr__(self) -> str:
        str_ = "Road[ \n"
        for dir_, lane in self.lanes.items():
            str_ += " {} [{}] \n".format(dir_, lane)
        str_ += "]"
        return str_
