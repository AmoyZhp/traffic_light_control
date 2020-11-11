from typing import Dict, List


from envs.phase import Direction, Movement, Phase


class Intersection():
    def __init__(self, id: str, phase_plan: List[Phase],
                 init_phase_index: int = 0) -> None:
        self.id = id
        self.phase_plan = phase_plan
        self.current_phase_index = init_phase_index
        self.roads: Dict[Direction, List] = {
            Direction.W: [],
            Direction.E: [],
            Direction.N: [],
            Direction.S: [], }

    def get_current_phase_index(self) -> int:
        return self.current_phase_index

    def get_current_phase(self) -> Phase:
        pass

    def get_roads_capacity(self, movement: Movement,
                           direction: Direction):
        pass

    def move_to_next_phase(self):
        self.current_phase_index = (
            self.current_phase_index + 1) % len(self.phase_plan)

    def get_lane_vechile(self, movement: Movement,
                         direction: Direction) -> int:
        pass
