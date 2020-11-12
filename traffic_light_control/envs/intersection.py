import cityflow
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
        capacity = road.get_capacity(streamDirection)
        return capacity

    def get_road_vehicles(self, loc: Location, graphDir: GraphDirection,
                          streamDirection: TrafficStreamDirection
                          ) -> int:
        road = self.roads[loc][graphDir]
        vehicles = road.get_vehicles(streamDirection)
        return vehicles

    def get_road_waiting_vehicles(self, loc: Location,
                                  graphDir: GraphDirection,
                                  streamDirection: TrafficStreamDirection
                                  ) -> int:
        road = self.roads[loc][graphDir]
        vehicles = road.get_vehicles(streamDirection)
        return vehicles

    def move_to_next_phase(self):
        self.current_phase_index = (
            self.current_phase_index + 1) % len(self.phase_plan)
