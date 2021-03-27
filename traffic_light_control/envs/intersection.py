from envs.lane import Lane
import numpy as np
from envs.road import Road
from typing import Dict, List
from envs.enum import Movement, Stream

PHASE_SPACE = 12

ROAD_STATE = True


class RoadLink():
    def __init__(
        self,
        start_road: Road,
        end_road: Road,
        lane_links: List[List[str]],
        movement: Movement,
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


class Intersection():
    def __init__(
        self,
        id: str,
        phase_plan: List[List[int]],
        roadlinks: List[RoadLink],
        roads: Dict[str, Road],
        init_phase_index: int = 0,
    ) -> None:
        self.id = id
        self.phase_plan = phase_plan
        self.current_phase_index = init_phase_index
        self.roadlinks = roadlinks
        self.roads = roads
        self._validity()

        self.state_space = 0
        if ROAD_STATE:
            for road in self.roads.values():
                self.state_space += road.state_space
        else:
            for rlink in self.roadlinks:
                for road in rlink.values():
                    self.state_space += road.state_space
        # first phase space is current phase
        # second belong to next phase
        self.phase_space = len(self.roadlinks)
        self.state_space += self.phase_space

    def update(
        self,
        vehicles_data_pool,
        waiting_vehicles_data_pool,
    ):
        for road in self.roads.values():
            road.update(
                vehicles_data_pool=vehicles_data_pool,
                waiting_vehicles_data_pool=waiting_vehicles_data_pool,
            )

    def _validity(self):
        for rlink in self.roadlinks:
            start_road = rlink.start_road
            if start_road.id not in self.roads:
                raise ValueError(
                    "road %s and road link mistatched ",
                    start_road.id,
                )
            end_road = rlink.end_road
            if end_road.id not in self.roads:
                raise ValueError(
                    "road %s and road link mistatched ",
                    end_road.id,
                )

    def update_vehicles_dict(self, vehicles_dict: Dict):
        for road in self.roads:
            road.update_vehicles_dict(vehicles_dict)

    def get_roads(self):
        return self.roads

    def get_state_space(self):
        return self.state_space

    def get_id(self) -> str:
        return self.id

    def get_roadlinks_len(self) -> str:
        return len(self.roadlinks)

    def get_roads_ids(self):
        return list(self.roads.keys())

    def get_current_phase_index(self) -> int:
        return self.current_phase_index

    def get_current_phase(self) -> List[int]:
        return self.phase_plan[self.current_phase_index]

    def get_waiting_rate(self) -> float:
        waiting_rate = 0.0
        for road in self.roads.values():
            for stream in Movement:
                capacity = road.get_incoming_capacity(stream)
                if capacity == 0:
                    continue
                waiting_lane = road.get_incoming_waiting_vehicles(stream)
                density = waiting_lane / capacity
                waiting_rate += density
        return waiting_rate

    def get_pressure(self) -> float:
        pressure = 0.0
        for rlink in self.roadlinks:
            pressure += rlink.pressure
        pressure = abs(pressure)
        # scaling
        pressure /= len(self.roadlinks)
        return pressure

    def move_to_next_phase(self):
        self.current_phase_index = (self.current_phase_index + 1) % len(
            self.phase_plan)

    def set_phase_index(self, index):
        assert index >= 0 and index < len(self.phase_plan)
        self.current_phase_index = index

    def to_tensor(self) -> np.ndarray:
        if ROAD_STATE:
            tensor = self._road_state()
        else:
            tensor = self._road_link_state()

        current_phase = self.phase_plan[self.current_phase_index]

        current_phase_tensor = np.zeros(self.phase_space)
        for i in current_phase:
            current_phase_tensor[i] = 1

        tensor = np.hstack((tensor, current_phase_tensor))
        return tensor

    def _road_link_state(self):
        raise NotImplementedError

    def _road_state(self):
        tensor = np.array([], dtype=np.float)
        for id in sorted(self.roads.keys()):
            road = self.roads[id]
            tensor = np.hstack((tensor, road.tensor))
        return tensor