from typing import Dict, List
import cityflow
import numpy as np
from envs.lane import Lane
from envs.phase import TrafficStreamDirection


class Road():
    def __init__(self, id: str,
                 lanes: Dict[TrafficStreamDirection, List[Lane]],
                 eng: cityflow.Engine) -> None:
        self.id = id
        self.eng = eng
        self.lanes = lanes
        self.stream_capacity: Dict[TrafficStreamDirection, int] = {
            TrafficStreamDirection.STRAIGHT: 0,
            TrafficStreamDirection.LEFT: 0,
            TrafficStreamDirection.RIGHT: 0,
        }
        for direction, directed_lanes in self.lanes.items():
            for lane in directed_lanes:
                self.stream_capacity[direction] += lane.get_capacity()

    def get_capacity(self, streamDir: TrafficStreamDirection) -> int:
        if streamDir not in self.lanes.keys():
            return 0

        return self.stream_capacity[streamDir]

    def get_vehicles(self, streamDir: TrafficStreamDirection) -> int:
        if streamDir not in self.lanes.keys():
            return 0
        vehicles = 0
        directed_lanes = self.lanes[streamDir]
        vehicles_dict = self.eng.get_lane_vehicle_count()
        for lane in directed_lanes:
            if lane.get_id() not in vehicles_dict.keys():
                print("key error of lane id {}".format(lane.get_id()))
            vehicles += vehicles_dict[lane.get_id()]
        return vehicles

    def get_waiting_vehicles(self, streamDir: TrafficStreamDirection) -> int:
        if streamDir not in self.lanes.keys():
            return 0
        directed_lanes = self.lanes[streamDir]
        vehicles = 0
        vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        for lane in directed_lanes:
            if lane.get_id() in vehicles_dict.keys():
                vehicles += vehicles_dict[lane.get_id()]
        return vehicles

    def to_tensor(self) -> np.ndarray:
        tensor = np.zeros(3)
        dire = TrafficStreamDirection.LEFT
        tensor[0] = (0 if self.get_capacity(dire) == 0 else
                     self.get_vehicles(dire) / self.get_capacity(dire))

        dire = TrafficStreamDirection.STRAIGHT
        tensor[1] = (0 if self.get_capacity(dire) == 0 else
                     self.get_vehicles(dire) / self.get_capacity(dire))

        dire = TrafficStreamDirection.RIGHT
        tensor[2] = (0 if self.get_capacity(dire) == 0 else
                     self.get_vehicles(dire) / self.get_capacity(dire))
        return tensor
