import cityflow
import numpy as np
from numpy.core.defchararray import greater
from envs.road import Road
from typing import Dict, List


from envs.phase import GraphDirection
from envs.phase import Location, Phase, TrafficStreamDirection


class Intersection():
    def __init__(self, id: str, phase_plan: List[Phase],
                 roads: Dict[Location, Dict[GraphDirection, Road]],
                 init_phase_index: int = 0) -> None:
        self.id = id
        self.phase_plan = phase_plan
        self.current_phase_index = init_phase_index
        self.roads = roads

    def get_id(self) -> str:
        return self.id

    def get_current_phase_index(self) -> int:
        return self.current_phase_index

    def get_current_phase(self) -> Phase:
        return self.phase_plan[self.current_phase_index]

    def get_road_capacity(self, loc: Location, graphDir: GraphDirection,
                          streamDirection: TrafficStreamDirection,
                          ) -> int:
        road = self.roads[loc][graphDir]
        if road is None:
            return 0
        capacity = road.get_capacity(streamDirection)
        return capacity

    def get_road_vehicles(self, loc: Location, graphDir: GraphDirection,
                          streamDirection: TrafficStreamDirection
                          ) -> int:
        road = self.roads[loc][graphDir]
        if road is None:
            return 0
        vehicles = road.get_vehicles(streamDirection)
        return vehicles

    def get_road_waiting_vehicles(self, loc: Location,
                                  graphDir: GraphDirection,
                                  streamDirection: TrafficStreamDirection
                                  ) -> int:

        road = self.roads[loc][graphDir]
        if road is None:
            return 0
        vehicles = road.get_vehicles(streamDirection)
        return vehicles

    def move_to_next_phase(self):
        self.current_phase_index = (
            self.current_phase_index + 1) % len(self.phase_plan)

    def to_tensor(self) -> np.ndarray:
        tensor = np.array([])

        tensor = np.hstack(
            (tensor,
             self.roads[Location.W][GraphDirection.IN].to_tensor()))
        tensor = np.hstack(
            (tensor,
             self.roads[Location.W][GraphDirection.OUT].to_tensor()))

        tensor = np.hstack(
            (tensor, self.roads[Location.E][GraphDirection.IN].to_tensor()))
        tensor = np.hstack(
            (tensor, self.roads[Location.E][GraphDirection.OUT].to_tensor()))

        tensor = np.hstack(
            (tensor, self.roads[Location.N][GraphDirection.IN].to_tensor())
        )
        tensor = np.hstack(
            (tensor, self.roads[Location.N][GraphDirection.OUT].to_tensor())
        )

        tensor = np.hstack(
            (tensor, self.roads[Location.S][GraphDirection.IN].to_tensor())
        )
        tensor = np.hstack(
            (tensor, self.roads[Location.S][GraphDirection.OUT].to_tensor())
        )

        current_phase_one_hot = np.zeros(len(self.phase_plan))
        current_phase_one_hot[self.current_phase_index] = 1

        tensor = np.hstack(
            (tensor, current_phase_one_hot)
        )

        next_phase_index = (self.current_phase_index +
                            1) % len(self.phase_plan)
        next_phase_one_hot = np.zeros(len(self.phase_plan))
        next_phase_one_hot[next_phase_index] = 1

        tensor = np.hstack(
            (tensor, next_phase_one_hot)
        )
        return tensor
