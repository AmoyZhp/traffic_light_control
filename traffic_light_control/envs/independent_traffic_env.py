from typing import Dict
import gym
from envs.intersection import Intersection
import numpy as np


class IndependentTrafficEnv(gym.Env):
    def __init__(self, eng, id_: str, max_time: int, interval: int,
                 intersections: Dict[str, Intersection]):
        super().__init__()
        self.eng = eng
        self.id_ = id_
        self.intersections = intersections
        self.interval = interval
        self.max_time = max_time
        self.time = 0

    def step(self, actions: Dict[str, int]):
        for id_, act in actions.items():
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
        info = []
        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        self.eng.reset()
        self.time = 0
        return self.__compute_state()

    def intersection_ids(self):
        return self.intersections.keys()

    def __compute_state(self) -> Dict[str, np.ndarray]:
        # 基于 intersections 计算全局的 state 情况
        state = {}
        for id_, item in self.intersections.items():
            state[id_] = item.to_tensor()
        return state

    def __compute_reward(self) -> Dict[str, float]:
        reward = {}
        for id_, inter in self.intersections.items():
            r = - inter.get_waiting_rate()
            reward[id_] = r
        return reward
