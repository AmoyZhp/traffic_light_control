
class Action(object):
    def __init__(self, intersection_id: str, keep_phase: bool):
        self.intersection_id = intersection_id
        self.keep_phase = keep_phase

    def get_keep_phase(self) -> bool:
        return self.keep_phase

    def get_intersection_id(self) -> str:
        return self.intersection_id
