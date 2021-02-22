from typing import List

import numpy as np

from hprl.util.typing import Action, State, Transition
from hprl.policy.core import Policy


class EpsilonGreedy(Policy):
    """
        this class is a wrapper, usually to wrapping the q-learning like policy.
    """

    def __init__(self,
                 inner_policy: Policy,
                 eps_frame: int,
                 eps_init: float,
                 eps_min: float,
                 action_space) -> None:
        super().__init__()
        self.inner_policy = inner_policy
        self.step = 0
        self.eps_frame = eps_frame
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps = eps_init
        self.action_space = action_space

    def compute_action(self, state: State) -> Action:
        self.step = min(self.step + 1, self.eps_frame)
        self.eps = max(self.eps_init - self.step /
                       self.eps_frame, self.eps_min)
        if np.random.rand() < self.eps:
            action = np.random.choice(range(self.action_space), 1).item()
            return action
        return super().compute_action(state)

    def learn_on_batch(self, batch_data: List[Transition]):
        return super().learn_on_batch(batch_data)

    def get_weight(self):
        return super().get_weight()

    def set_weight(self, weight):
        return super().set_weight(weight)

    def unwrapped(self):
        return self.inner_policy
