from envs.road import Road
import logging
from hprl.util.typing import Action
from envs.intersection import Intersection
from typing import Dict, List, Tuple
import hprl
import numpy as np

logger = logging.getLogger(__name__)


class TrafficLightCtrlEnv(hprl.MultiAgentEnv):

    ACTION_SPACE = 2

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
        self.local_action_space = {}
        self.local_state_space = {}
        self.roads_set: Dict[str, Road] = {}
        for id, inter in self.intersections.items():
            self.local_state_space[id] = inter.get_state_space()
            self.local_action_space[id] = self.ACTION_SPACE
            roads = inter.get_roads()
            for r in roads.values():
                if r.id not in self.roads_set.keys():
                    self.roads_set[r.id] = r

        self.central_state_space = 0
        for road in self.roads_set.values():
            self.central_state_space += road.get_state_space()

        self.central_action_space = self.ACTION_SPACE**len(self.intersections)
        self._log_init_info()

    def _log_init_info(self):

        logger.info("Traffic Ligth Ctrl Env init info")
        logger.info("env name : %s", self.name)
        logger.info("max time : %d", self.max_time)
        logger.info("interval : %d", self.interval)
        logger.info("central state space : %d", self.central_state_space)
        logger.info("cnetral action space : %d", self.central_action_space)
        for id in self.intersections_id:
            s_space = self.intersections[id].get_state_space()
            logger.info("intersection %s", id)
            logger.info("\t local state space is %d", s_space)
            logger.info("\t local action space is %d", self.ACTION_SPACE)
        logger.info("roads set : %s", list(self.roads_set.keys()))

        logger.info("Traffic Ligth Ctrl Env init info end")

    def step(
        self, action: hprl.Action
    ) -> Tuple[hprl.State, hprl.Reward, hprl.Terminal, Dict]:
        self._process_action(action)
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
        info = self._get_info()
        return next_state, reward, terminal, info

    def _get_info(self):
        info = {"avg_travel_time": self.eng.get_average_travel_time()}
        return info

    def _process_action(self, action: Action):
        for id in self.intersections_id:
            act = action.local[id]
            if act == 1:
                # act = 1 表示要切换到下一个状态
                intersection = self.intersections[id]
                intersection.move_to_next_phase()
                self.eng.set_tl_phase(id, intersection.current_phase_index)

    def _compute_state(self) -> hprl.State:

        state = {}
        for id in self.intersections_id:
            item = self.intersections[id]
            state[id] = item.to_tensor()
        roads = []
        for r in self.roads_set.values():
            roads.append(r.to_tensor())
        central_state = np.hstack(roads)
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

    @property
    def setting(self):
        _setting = {
            "max_time": self.max_time,
            "interval": self.interval,
            "env_name": self.get_env_name(),
            "agents_id": self.get_agents_id(),
            "central_state_space": self.get_central_state_space(),
            "central_action_space": self.get_central_action_space(),
            "local_action_space": self.get_local_action_space(),
            "local_state_space": self.get_local_state_space(),
        }
        return _setting


class PhaseChosenEnv(TrafficLightCtrlEnv):
    def __init__(
        self,
        eng,
        name: str,
        max_time: int,
        interval: int,
        intersections: Dict[str, Intersection],
    ):
        super().__init__(
            eng,
            name,
            max_time,
            interval,
            intersections,
        )

    def _process_action(self, action: Action):
        return super()._process_action(action)

    def _get_info(self):
        return super()._get_info()
