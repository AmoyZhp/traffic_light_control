from typing import Dict, List, Tuple

import gym
import numpy as np

from hprl.env.multi_agent_env import MultiAgentEnv
from hprl.util.typing import Action, Reward, State, Terminal


class GymWrapper(MultiAgentEnv):
    def __init__(self, env: gym.Env) -> None:
        super().__init__()
        self.env = env
        self.local_ids = ["1"]

    def step(self, actions: Action) -> Tuple[State, Reward, Terminal, Dict]:

        action = actions.local[self.local_ids[0]]

        s, r, done, info = self.env.step(action)

        state = State(local={self.local_ids[0]: np.array(s)})
        reward = Reward(local={self.local_ids[0]: r})
        termial = Terminal(central=done, local={self.local_ids[0]: done})

        return state, reward, termial, info

    def reset(self) -> State:
        state = self.env.reset()
        return State(local={self.local_ids[0]: np.array(state)})

    def get_agents_id(self) -> List[str]:
        return self.local_ids

    def get_central_action_space(self):
        ...

    def get_local_action_space(self):
        ...

    def get_local_state_space(self):
        ...

    def get_central_state_space(self):
        ...

    def get_env_name(self):
        ...