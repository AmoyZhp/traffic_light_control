from typing import Dict, List, Tuple

import gym
import numpy as np

from hprl.env.core import MultiAgentEnv
from hprl.util.typing import Action, Reward, State


class GymWrapper(MultiAgentEnv):
    def __init__(self, env: gym.Env) -> None:
        super().__init__()
        self.env = env
        self.local_ids = ["1"]

    def step(self,
             actions: Action
             ) -> Tuple[State, Reward, bool, Dict]:

        action = actions.local[self.local_ids[0]]

        s, r, done, info = self.env.step(action)

        state = State(local={
            self.local_ids[0]: np.array(s)
        }),
        reward = Reward(
            local={
                self.local_ids[0]: r,
            }
        ),
        return state, reward, done, info

    def reset(self) -> State:
        state = self.env.reset()
        return State(local={
            self.local_ids[0]: np.array(state)
        })

    def get_agents_id(self) -> List[str]:
        return self.local_ids
