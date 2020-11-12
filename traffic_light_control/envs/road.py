from typing import Dict, List
import cityflow
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
        return self.stream_capacity[streamDir]

    def get_vehicles(self, streamDir: TrafficStreamDirection) -> int:
        directed_lanes = self.lanes[streamDir]
        vehicles = 0
        vehicles_dict = self.eng.get_lane_vehicle_count()
        for lane in directed_lanes:
            vehicles += vehicles_dict[lane.get_id()]
        return vehicles

    def get_waiting_vehicles(self, streamDir: TrafficStreamDirection) -> int:
        directed_lanes = self.lanes[streamDir]
        vehicles = 0
        vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        for lane in directed_lanes:
            vehicles += vehicles_dict[lane.get_id()]
        return vehicles
