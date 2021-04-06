from typing import Dict, List, Tuple

import gym
import numpy as np
from hprl.env.multi_agent import MultiAgentEnv
from hprl.typing import Action, Reward, State, Terminal


class GymWrapper(MultiAgentEnv):
    def __init__(self, env: gym.Env) -> None:
        self._env = env
        self._agents_id = ["1"]

    def step(self, actions: Action) -> Tuple[State, Reward, Terminal, Dict]:

        action = actions.local[self._agents_id[0]]

        s, r, done, info = self._env.step(action)

        state = State(local={self._agents_id[0]: np.array(s)})
        reward = Reward(central=0.0, local={self._agents_id[0]: r})
        termial = Terminal(central=done, local={self._agents_id[0]: done})

        return state, reward, termial, info

    def reset(self) -> State:
        state = self._env.reset()
        return State(local={self._agents_id[0]: np.array(state)})

    @property
    def agents_id(self) -> List[str]:
        return self._agents_id

    @property
    def id(self):
        ...

    @property
    def setting(self):
        ...

    @property
    def central_action_space(self):
        ...

    @property
    def local_action_space(self):
        ...

    @property
    def central_state_space(self):
        ...

    @property
    def local_state_space(self):
        ...
