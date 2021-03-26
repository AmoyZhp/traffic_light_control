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
        self.lane_links = lane_links
        self.movement = movement

    @property
    def start_road(self):
        return self._start_road

    @property
    def end_road(self):
        return self._end_road

    def cal_pressure(self):
        pressure = 0.0
        for lane_link in self.lane_links:
            pressure += self._cal_lane_link_pressure(lane_link)
        pressure /= len(self.lane_links)
        return pressure

    def _cal_lane_link_pressure(self, lane_link: List[str]):
        assert len(lane_link) == 2
        start_lane_id = lane_link[0]
        end_lane_id = lane_link[1]
        incoming_lane = self.start_road.get_lane(start_lane_id)
        outgoing_lane = self.end_road.get_lane(end_lane_id)
        pressure = incoming_lane.get_density() - outgoing_lane.get_density()
        return pressure


class Intersection():
    def __init__(
        self,
        id: str,
        phase_plan: List[List[int]],
        roadlinks: List[RoadLink],
        init_phase_index: int = 0,
    ) -> None:
        self.id = id
        self.phase_plan = phase_plan
        self.current_phase_index = init_phase_index
        self.roadlinks = roadlinks
        self.roads: Dict[str, Road] = {}

        for rlink in self.roadlinks:
            start_road = rlink.start_road
            if start_road.id not in self.roads:
                self.roads[start_road.id] = start_road
            end_road = rlink.end_road
            if end_road.id not in self.roads:
                self.roads[end_road.id] = end_road

        self.state_space = 0
        if ROAD_STATE:
            for road in self.roads.values():
                self.state_space += road.get_state_space()
        else:
            for rlink in self.roadlinks:
                for road in rlink.values():
                    self.state_space += road.get_state_space()
        # first phase space is current phase
        # second belong to next phase
        self.phase_space = len(self.roadlinks)
        self.state_space += self.phase_space

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
                capacity = road.get_capacity(stream)
                if capacity == 0:
                    continue
                waiting_lane = road.get_waiting_vehicles(stream)
                density = waiting_lane / capacity
                waiting_rate += density
        return waiting_rate

    def get_pressure(self) -> float:
        pressure = 0.0
        cnt = 0
        for rlink in self.roadlinks:
            road_press = 0.0
            incoming_road = rlink[Stream.IN]
            outgoing_road = rlink[Stream.OUT]
            # because vehicle can change lane in outgoing road
            # so outgoring density is the road density
            out_road_density = outgoing_road.get_density()
            assert out_road_density <= 1 and out_road_density >= 0
            for direction in Movement:
                if (incoming_road.get_capacity(direction) == 0):
                    continue
                in_density = incoming_road.get_vehicles(
                    direction) / incoming_road.get_capacity(direction)
                movement_press = in_density - out_road_density
                assert movement_press <= 1 and movement_press >= -1
                road_press += movement_press
                cnt += 1

            pressure += road_press
        pressure = abs(pressure)
        # scaling
        pressure /= cnt
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
        tensor = np.array([], dtype=np.float)
        for rlink in self.roadlinks:
            in_road = rlink[Stream.IN]
            tensor = np.hstack((tensor, in_road.to_tensor()))
            out_road = rlink[Stream.OUT]
            tensor = np.hstack((tensor, out_road.to_tensor()))
        return tensor

    def _road_state(self):
        tensor = np.array([], dtype=np.float)
        for id in sorted(self.roads.keys()):
            road = self.roads[id]
            tensor = np.hstack((tensor, road.to_tensor()))
        return tensor