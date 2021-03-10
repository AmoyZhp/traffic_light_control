from hprl.util.typing import Action
from trainer.independent_traffic_trainer import ACTION_SPACE
from envs.intersection import Intersection
from typing import Dict, List, Tuple
import hprl
import numpy as np


class TrafficLightCtrlEnv(hprl.MultiAgentEnv):
    def __init__(
        self,
        eng,
        name: str,
        max_time: int,
        interval: int,
        intersections: Dict[str, Intersection],
    ):
        super().__init__()
        self.eng = eng
        self.name = name
        self.intersections = intersections
        self.intersections_id = sorted(list(self.intersections.keys()))
        self.interval = interval
        self.max_time = max_time
        self.time = 0
        self.central_state_space = 0
        self.central_action_space = len(self.intersections_id) * ACTION_SPACE
        self.local_action_space = {}
        self.local_state_space = {}
        for id in self.intersections_id:
            s_space = self.intersections[id].get_state_space()
            self.central_state_space += s_space
            self.local_state_space[id] = s_space
            self.local_action_space[id] = ACTION_SPACE

    def step(
        self, action: hprl.Action
    ) -> Tuple[hprl.State, hprl.Reward, hprl.Terminal, Dict]:

        for id in self.intersections_id:
            act = action.local[id]
            if act == 1:
                # act = 1 表示要切换到下一个状态
                intersection = self.intersections[id]
                intersection.move_to_next_phase()
                self.eng.set_tl_phase(id, intersection.current_phase_index)
        for _ in range(self.interval):
            self.eng.next_step()

        self.time += self.interval
        next_state = self._compute_state()
        reward = self._compute_reward()
        done = False if self.time < self.max_time else True
        terminal = hprl.Terminal(
            central=done,
            local={id: done
                   for id in self.intersections_id},
        )
        info = {"avg_travel_time": self.eng.get_average_travel_time()}
        return next_state, reward, terminal, info

    def _compute_state(self) -> hprl.State:

        state = {}
        central_state = []
        for id in self.intersections_id:
            item = self.intersections[id]
            state[id] = item.to_tensor()
            central_state.append(item.to_tensor())
        central_state = np.hstack(central_state)

        state = hprl.State(central=central_state, local=state)
        return state

    def _compute_reward(self) -> hprl.Reward:
        central_reward = 0.0
        local_reward = {}
        for id in self.intersections_id:
            intersection = self.intersections[id]
            # r = - inter.get_waiting_rate()
            r = -intersection.get_pressure()
            local_reward[id] = r
            central_reward += r
        central_reward /= len(self.intersections)
        reward = hprl.Reward(central=central_reward, local=local_reward)
        return reward

    def reset(self) -> np.ndarray:
        self.eng.reset()
        self.time = 0
        return self._compute_state()

    def get_central_action_space(self):
        return self.central_action_space

    def get_local_action_space(self):
        return self.local_action_space

    def get_central_state_space(self):
        return self.central_state_space

    def get_local_state_space(self):
        return self.local_state_space

    def get_agents_id(self) -> List[str]:
        return self.intersections_id

    def get_env_name(self):
        return self.name