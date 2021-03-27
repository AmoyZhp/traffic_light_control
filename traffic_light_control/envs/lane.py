from typing import Dict
from envs.enum import IncomingDirection


class Lane():
    def __init__(
        self,
        id: str,
        belonged_road: str,
        capacity: int,
        incoming_type: IncomingDirection,
    ) -> None:
        self._id = id
        self._belonged_road = belonged_road
        self._capacity = capacity
        self._incoming_type = incoming_type
        self._vehicles = 0
        self._waiting_vehicles = 0

    def update(
        self,
        vehicles_data_pool,
        waiting_vehicles_data_pool,
    ):
        self._vehicles = vehicles_data_pool[self._id]
        self._waiting_vehicles = waiting_vehicles_data_pool[self._id]

    @property
    def density(self):
        return self._vehicles / self._capacity

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
    def vehicles(self):
        return self._vehicles

    @property
    def waiting_vehicles(self):
        return self._waiting_vehicles

    def __repr__(self) -> str:
        return "Lane[ id {} , capacity {} ]".format(self._id, self._capacity)
