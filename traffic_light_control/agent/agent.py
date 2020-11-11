import random
from basis.action import Action
from basis.state import State


class Agent():

    def __init__(self, intersection_id: str):
        """init agent

        Args:
            intersection_id (str): intersection id of agent
        """

        self.intersection_id = intersection_id

    def act(self, state: State) -> Action:
        """agent acting

         Args:
             state (State): Traffic State

         Returns:
             Action: action
        """
        temp = random.randint(0, 1)
        keep_phase = True
        if temp == 0:
            keep_phase = False
        action = Action(self.intersection_id, keep_phase)
        return action
