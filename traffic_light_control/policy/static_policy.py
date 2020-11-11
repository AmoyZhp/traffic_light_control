from typing import List


class StaticPolicy():

    def __init__(self) -> None:
        pass

    def act(self, current_time: int, phase_plan: List[int]) -> bool:
        time = current_time % (phase_plan[-1] + 1)
        for t in phase_plan:
            if time == t:
                return False
        return True
