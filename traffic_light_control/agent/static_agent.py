from typing import List


from basis.action import Action


class StaticAgent():
    def __init__(self, intersection_id: str, phase_plan: List[int]) -> None:
        self.intersection_id = intersection_id
        self.phase_plan = phase_plan

    def act(self, current_time: int) -> Action:
        time = current_time % (self.phase_plan[-1] + 1)
        for t in self.phase_plan:
            if time == t:
                return Action(self.intersection_id, False)
        return Action(self.intersection_id, True)
