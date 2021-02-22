from . import Intersection
from typing import Dict
import hprl
import numpy as np


class TrafficLightCtrlEnv(hprl.MultiAgentEnv):

    OBS_SPACE = 6*4 + 12*2
    ACTION_SPACE = 2

    def __init__(self,
                 eng,
                 id_: str,
                 max_time: int,
                 interval: int,
                 intersections: Dict[str, Intersection]):
        super().__init__()
        self.eng = eng
        self.id_ = id_
        self.intersections = intersections
        self.loacal_ids = sorted(list(self.intersections.keys()))
        self.interval = interval
        self.max_time = max_time
        self.time = 0

    def step(self, actions: hprl.Action) -> hprl.EnvReturn:
        local_actions = actions.local
        for id_, act in local_actions.items():
            if act == 1:
                # act = 1 表示要切换到下一个状态
                intersection = self.intersections[id_]
                intersection.move_to_next_phase()
                self.eng.set_tl_phase(
                    id_, intersection.current_phase_index)
        for _ in range(self.interval):
            self.eng.next_step()
        self.time += self.interval
        next_state = self.__compute_state()
        reward = self.__compute_reward()
        done = False if self.time < self.max_time else True
        info = {
            "average_travel_time": self.eng.get_average_travel_time()
        }
        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        self.eng.reset()
        self.time = 0
        return self.__compute_state()

    def intersection_ids(self):
        return self.loacal_ids

    def __compute_state(self) -> Dict[str, np.ndarray]:
        """
            状态由两个部分组成，中心和局部。
            局部状态是每个路口的状态
            中心状态是路口状态的拼接

        """
        state = {}
        central_state = []
        for id_ in self.loacal_ids:
            item = self.intersections[id_]
            state[id_] = item.to_tensor()
            central_state.append(item.to_tensor())
        central_state = np.hstack(central_state)
        state = hprl.State(
            central_state=central_state,
            local=state
        )
        return state

    def __compute_reward(self) -> Dict[str, float]:
        central_reward = 0.0
        local_reward = {}
        for id_, inter in self.intersections.items():
            # r = - inter.get_waiting_rate()
            r = - inter.get_pressure()
            local_reward[id_] = r
            central_reward += r
        central_reward /= len(self.intersections)

    def get_action_space(self):
        return self.ACTION_SPACE

    def get_local_obs_space(self):
        return self.OBS_SPACE

    def get_state_space(self):
        return self.OBS_SPACE * len(self.loacal_ids)
