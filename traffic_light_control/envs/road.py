from typing import Dict, List
import cityflow
import numpy as np
from envs.lane import Lane
from envs.enum import Movement


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

        self._incomings_capacity: Dict[Movement, int] = {
            Movement.STRAIGHT: 0,
            Movement.LEFT: 0,
            Movement.RIGHT: 0,
        }

        self._capacity = 0
        for lane in self._lanes.values():
            self._capacity += lane._capacity
            if lane.incoming_type is not None:
                self._incomings_capacity[lane.incoming_type] += lane._capacity
        self._state_space = len(self._lanes.keys())

        self._vehicles = 0
        self._waiting_vehicles = 0
        self._incoming_vehicles: Dict[Movement, int] = {}
        self._incoming_watiting_vehicles: Dict[Movement, int] = {}

    def update(
        self,
        vehicles_data_pool,
        waiting_vehicles_data_pool,
    ):
        for lane in self._lanes.values():
            lane.update(
                vehicles_data_pool=vehicles_data_pool,
                waiting_vehicles_data_pool=waiting_vehicles_data_pool,
            )
        self._vehicles = 0
        self._waiting_vehicles = 0
        for incoming_type in Movement:
            self._incoming_vehicles[incoming_type] = 0
            self._incoming_watiting_vehicles[incoming_type] = 0
        for lane in self._lanes.values():
            self._vehicles += lane.vehicles
            self._waiting_vehicles += lane._waiting_vehicles
            if lane._incoming_type is not None:
                self._incoming_vehicles[lane.incoming_type] += lane.vehicles
                self._incoming_watiting_vehicles[
                    lane.incoming_type] += lane.waiting_vehicles

    def get_incoming_capacity(self, stream_dir: Movement) -> int:
        return self._incomings_capacity[stream_dir]

    def get_incoming_vehicles(self, stream_dir: Movement) -> int:
        return self._incoming_vehicles[stream_dir]

    def get_incoming_waiting_vehicles(self, stream_dir: Movement) -> int:
        return self._incoming_watiting_vehicles[stream_dir]

    def get_lane(self, id):
        return self._lanes[id]

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

    def __repr__(self) -> str:
        str_ = "Road[ \n"
        for dir_, lane in self._lanes.items():
            str_ += " {} [{}] \n".format(dir_, lane)
        str_ += "]"
        return str_
