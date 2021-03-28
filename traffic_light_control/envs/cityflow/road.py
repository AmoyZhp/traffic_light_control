from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List
import numpy as np


class IncomingDirection(Enum):
    STRAIGHT = "go_straight"
    LEFT = "turn_left"
    RIGHT = "turn_right"


class Stream(Enum):
    IN = auto()
    OUT = auto()


@dataclass
class Vehicle():
    speed: float
    distance: float
    drivable: str
    road: str
    intersection: str
    route: str


class Lane():
    def __init__(
        self,
        id: str,
        belonged_road: str,
        capacity: int,
        incoming_type: IncomingDirection,
        max_speed: float,
    ) -> None:
        self._id = id
        self._belonged_road = belonged_road
        self._capacity = capacity
        self._incoming_type = incoming_type
        self._max_speed = max_speed
        self._vehicles_count = 0
        self._waiting_vehicles_count = 0
        self._vehicles: Dict[str, Vehicle] = {}
        self._vehicles_speed: Dict[str, float] = {}

    def update(
        self,
        vehicles_data_pool,
        waiting_vehicles_data_pool,
        vehicles_speed_pool,
    ):
        self._vehicles_count = vehicles_data_pool[self._id]
        self._waiting_vehicles_count = waiting_vehicles_data_pool[self._id]
        self._vehicles_speed = vehicles_speed_pool[self._id]

    @property
    def avg_speed_rate(self):
        if not self._vehicles_speed:
            return 0.0
        _avg_speed_rate = self.avg_speed_sum / len(self._vehicles_speed)
        return _avg_speed_rate

    @property
    def avg_speed_sum(self):
        _avg_speed = 0.0
        for speed in self._vehicles_speed.values():
            _avg_speed += speed / self._max_speed
        return _avg_speed

    @property
    def waiting_rate(self):
        return self._waiting_vehicles_count / self._vehicles_count

    @property
    def density(self):
        return self._vehicles_count / self._capacity

    @property
    def id(self):
        return self._id

    @property
    def belonged_road(self):
        return self._belonged_road

    @property
    def capacity(self):
        return self._capacity

    @property
    def incoming_type(self):
        return self._incoming_type

    @property
    def vehicles_count(self):
        return self._vehicles_count

    @property
    def waiting_vehicles_count(self):
        return self._waiting_vehicles_count

    @property
    def max_speed(self):
        return self._max_speed


class Road():
    def __init__(
        self,
        id: str,
        lanes: Dict[str, Lane],
        start: str,
        end: str,
    ) -> None:
        self._id = id
        self._lanes = lanes
        self._start = start
        self._end = end

        self._incomings_capacity: Dict[IncomingDirection, int] = {
            IncomingDirection.STRAIGHT: 0,
            IncomingDirection.LEFT: 0,
            IncomingDirection.RIGHT: 0,
        }

        self._capacity = 0
        for lane in self._lanes.values():
            self._capacity += lane._capacity
            if lane.incoming_type is not None:
                self._incomings_capacity[lane.incoming_type] += lane._capacity
        self._state_space = len(self._lanes.keys())

        self._vehicles = 0
        self._waiting_vehicles = 0
        self._incoming_vehicles: Dict[IncomingDirection, int] = {}
        self._incoming_watiting_vehicles: Dict[IncomingDirection, int] = {}

    def update(
        self,
        vehicles_data_pool,
        waiting_vehicles_data_pool,
        vehicles_speed_pool,
    ):
        for lane in self._lanes.values():
            lane.update(
                vehicles_data_pool=vehicles_data_pool,
                waiting_vehicles_data_pool=waiting_vehicles_data_pool,
                vehicles_speed_pool=vehicles_speed_pool,
            )
        self._vehicles = 0
        self._waiting_vehicles = 0
        for incoming_type in IncomingDirection:
            self._incoming_vehicles[incoming_type] = 0
            self._incoming_watiting_vehicles[incoming_type] = 0
        for lane in self._lanes.values():
            self._vehicles += lane.vehicles_count
            self._waiting_vehicles += lane._waiting_vehicles_count
            if lane._incoming_type is not None:
                self._incoming_vehicles[
                    lane.incoming_type] += lane.vehicles_count
                self._incoming_watiting_vehicles[
                    lane.incoming_type] += lane.waiting_vehicles_count

    def get_incoming_capacity(self, stream_dir: IncomingDirection) -> int:
        return self._incomings_capacity[stream_dir]

    def get_incoming_vehicles(self, stream_dir: IncomingDirection) -> int:
        return self._incoming_vehicles[stream_dir]

    def get_incoming_waiting_vehicles(self,
                                      stream_dir: IncomingDirection) -> int:
        return self._incoming_watiting_vehicles[stream_dir]

    def get_lane(self, id):
        return self._lanes[id]

    @property
    def avg_speed_rate(self):
        avg_speed_rate = self.avg_speed_sum / len(self._lanes)
        return avg_speed_rate

    @property
    def avg_speed_sum(self):
        _avg_speed_sum = 0.0
        for lane in self._lanes.values():
            _avg_speed_sum += lane.avg_speed_sum
        return _avg_speed_sum

    @property
    def id(self):
        return self._id

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def capacity(self):
        return self._capacity

    @property
    def vehicles(self):
        return self._vehicles

    @property
    def waiting_vehicles(self):
        return self._waiting_vehicles

    @property
    def density(self) -> int:
        return self._vehicles / self._capacity

    @property
    def state_space(self):
        return self._state_space

    @property
    def tensor(self) -> np.ndarray:
        vec = []
        for lane in self._lanes.values():
            vec.append(lane.density)
        _tensor = np.array(vec, dtype=np.float)
        assert _tensor.shape[0] == self._state_space
        return _tensor


class RoadLink():
    def __init__(
        self,
        start_road: Road,
        end_road: Road,
        lane_links: List[List[str]],
        movement: IncomingDirection,
    ) -> None:
        self._start_road = start_road
        self._end_road = end_road
        self._lane_links = lane_links
        self._in_direction = movement

    @property
    def start_road(self):
        return self._start_road

    @property
    def end_road(self):
        return self._end_road

    @property
    def in_direction(self):
        return self._in_direction

    @property
    def lane_links(self):
        return self._lane_links

    @property
    def pressure(self):
        start_lanes: Dict[str, Lane] = {}
        end_lanes: Dict[str, Lane] = {}
        for lane_link in self._lane_links:
            assert len(lane_link) == 2
            start_lane_id = lane_link[0]
            end_lane_id = lane_link[1]
            start_lanes[start_lane_id] = self.start_road.get_lane(
                start_lane_id)
            end_lanes[end_lane_id] = self.end_road.get_lane(end_lane_id)

        incoming_density = 0.0
        for lane in start_lanes.values():
            incoming_density += lane.density
        incoming_density /= len(start_lanes)

        outgoing_density = 0.0
        for lane in end_lanes.values():
            outgoing_density += lane.density
        outgoing_density /= len(end_lanes)

        _pressure = incoming_density - outgoing_density
        return _pressure
