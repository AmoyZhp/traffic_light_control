
class State():
    def __init__(self, vehicles_on_lanes: dict,
                 current_phase: int, next_phase: int):
        self.vehicles_on_lanes = vehicles_on_lanes
        self.current_phase = current_phase
        self.next_phase = next_phase

    def to_vector(self):
        pass
