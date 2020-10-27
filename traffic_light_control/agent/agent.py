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
        traffic_phase = random.randrange(0, 8)
        action = Action(self.intersection_id, traffic_phase)
        return action
