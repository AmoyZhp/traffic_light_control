from envs.road import Road
import logging
from hprl.util.typing import Action
from envs.intersection import Intersection, RoadLink
from typing import Dict, List, Tuple
import hprl
import numpy as np

logger = logging.getLogger(__name__)


class CityFlow(hprl.MultiAgentEnv):
    def __init__(
        self,
        eng,
        name: str,
        max_time: int,
        interval: int,
        intersections: Dict[str, Intersection],
    ) -> None:
        self._eng = eng
        self._name = name
        self._intersections = intersections
        self._intersections_id = list(self._intersections.keys())
        self._interval = interval
        self._max_time = max_time
        self._time = 0

        self._roads: Dict[str, Road] = {}
        for inter in self._intersections.values():
            roads = inter.get_roads()
            for r in roads.values():
                if r.id not in self._roads.keys():
                    self._roads[r.id] = r

        local_sp, local_ap = self._compute_local_space()
        self._local_state_space = local_sp
        self._local_action_space = local_ap
        self._central_state_space = self._compute_central_state_space()
        self._central_action_space = self._compute_cental_action_space()

        self._update_state()
        self._log_init_info()

    def step(
        self, action: hprl.Action
    ) -> Tuple[hprl.State, hprl.Reward, hprl.Terminal, Dict]:

        self._eng_step(action)
        next_state = self._compute_state()
        reward = self._compute_reward()
        terminal = self._is_terminal()
        info = self._get_info()
        return next_state, reward, terminal, info

    def reset(self) -> np.ndarray:
        self._eng.reset()
        self._update_state()
        self.time = 0
        return self._compute_state()

    def _eng_step(self, action: hprl.Action):
        for id in self._intersections_id:
            act = action.local[id]
            self._eng.set_tl_phase(id, act)
        for _ in range(self._interval):
            self._eng.next_step()
        self._update_state()
        self._time += self._interval

    def _update_state(self):
        vehicles_dict = self._eng.get_lane_vehicle_count()
        waiting_vehicles_dict = self._eng.get_lane_waiting_vehicle_count()
        for inter in self._intersections.values():
            inter.update(
                vehicles_data_pool=vehicles_dict,
                waiting_vehicles_data_pool=waiting_vehicles_dict,
            )

    def _compute_state(self) -> hprl.State:
        local_state = {}
        for id, inter in self._intersections.items():
            local_state[id] = inter.to_tensor()

        roads = []
        for r in self._roads.values():
            roads.append(r.tensor)
        central_state = np.hstack(roads)

        state = hprl.State(central=central_state, local=local_state)
        return state

    def _compute_reward(self) -> hprl.Reward:
        central_reward = 0.0
        local_reward = {}

        for id, inter in self._intersections.items():
            r = -inter.get_pressure()
            local_reward[id] = r
            central_reward += r

        central_reward /= len(self._intersections)

        reward = hprl.Reward(central=central_reward, local=local_reward)
        return reward

    def _is_terminal(self):
        done = False if self._time < self._max_time else True
        terminal = hprl.Terminal(
            central=done,
            local={id: done
                   for id in self._intersections_id},
        )
        return terminal

    def _get_info(self):
        info = {"avg_travel_time": self._eng.get_average_travel_time()}
        return info

    def _compute_central_state_space(self):
        central_state_space = 0
        for road in self._roads.values():
            central_state_space += road.state_space
        return central_state_space

    def _compute_cental_action_space(self):
        central_action_space = 1
        for a_space in self.local_action_space.values():
            central_action_space *= a_space
        return central_action_space

    def _compute_local_space(self):
        local_state_space = {}
        local_action_space = {}
        for id, iter in self._intersections.items():
            local_state_space[id] = iter.get_state_space()
            local_action_space[id] = len(iter.phase_plan)
        return local_state_space, local_action_space

    def _log_init_info(self):
        logger.info("City Flow Env init info : ")
        logger.info("env name : %s", self._name)
        logger.info("max time : %d", self._max_time)
        logger.info("interval : %d", self._interval)
        logger.info("central state space : %d", self._central_state_space)
        logger.info("cnetral action space : %d", self._central_action_space)
        for id in self._intersections_id:
            logger.info("intersection %s", id)
            logger.info(
                "\t local state space is %d",
                self._local_state_space[id],
            )
            logger.info(
                "\t local action space is %d",
                self._local_action_space[id],
            )

        logger.info("City Flow Env info log end")

    @property
    def setting(self):
        _setting = {
            "max_time": self._max_time,
            "interval": self._interval,
            "name": self._name,
            "agents_id": self.agents_id,
            "central_state_space": self._central_state_space,
            "central_action_space": self._central_action_space,
            "local_action_space": self._local_action_space,
            "local_state_space": self._local_state_space,
            "roads": str(self._roads.keys()),
        }
        return _setting

    @property
    def central_action_space(self):
        return self._central_action_space

    @property
    def local_action_space(self):
        return self._local_action_space

    @property
    def central_state_space(self):
        return self._central_state_space

    @property
    def local_state_space(self):
        return self._local_state_space

    @property
    def agents_id(self) -> List[str]:
        return self._intersections_id

    @property
    def name(self):
        return self._name


class MaxPressure(CityFlow):
    def __init__(
        self,
        eng,
        name: str,
        max_time: int,
        interval: int,
        intersections: Dict[str, Intersection],
    ) -> None:
        super().__init__(eng, name, max_time, interval, intersections)

    def _compute_state(self) -> hprl.State:
        local_state = {}
        for id in self._intersections_id:
            item = self._intersections[id]
            phase_plans: List[List[int]] = item.phase_plan
            roadlinks: List[RoadLink] = item.roadlinks
            phases_pressure = []
            for phase in phase_plans:
                pressure = 0.0
                for rlink_index in phase:
                    rlink = roadlinks[rlink_index]
                    pressure += rlink.pressure
                pressure = abs(pressure)
                phases_pressure.append(pressure)
            local_state[id] = phases_pressure

        state = hprl.State(local=local_state)
        return state

    def _log_init_info(self):
        ...


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
        self.roads_set: Dict[str, Road] = {}
        for inter in self.intersections.values():
            roads = inter.get_roads()
            for r in roads.values():
                if r.id not in self.roads_set.keys():
                    self.roads_set[r.id] = r
        local_sp, local_ap = self._compute_local_space()

        self.local_state_space = local_sp
        self.local_action_space = local_ap

        self.central_state_space = 0
        for road in self.roads_set.values():
            self.central_state_space += road.get_state_space()
        self.central_action_space = 1
        for a_space in self.local_action_space.values():
            self.central_action_space *= a_space

        self._log_init_info()

    def _compute_local_space(self):
        local_action_space = {}
        local_state_space = {}
        for id, inter in self.intersections.items():
            local_state_space[id] = inter.get_state_space()
            local_action_space[id] = self.ACTION_SPACE
        return local_state_space, local_action_space

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

    def _log_init_info(self):

        logger.info("Traffic Ligth Ctrl Env init info")
        logger.info("env name : %s", self.name)
        logger.info("max time : %d", self.max_time)
        logger.info("interval : %d", self.interval)
        logger.info("central state space : %d", self.central_state_space)
        logger.info("cnetral action space : %d", self.central_action_space)
        for id in self.intersections_id:
            logger.info("intersection %s", id)
            logger.info(
                "\t local state space is %d",
                self.local_state_space[id],
            )
            logger.info(
                "\t local action space is %d",
                self.local_action_space[id],
            )
        logger.info("roads set : %s", list(self.roads_set.keys()))

        logger.info("Traffic Ligth Ctrl Env init info end")


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
        for id in self.intersections_id:
            phase_index = action.local[id]
            self.eng.set_tl_phase(id, phase_index)
            self.intersections[id].set_phase_index(phase_index)

    def _compute_local_space(self):
        local_state_space = {}
        local_action_space = {}
        for id, iter in self.intersections.items():
            local_state_space[id] = iter.get_state_space()
            local_action_space[id] = len(iter.phase_plan)
        return local_state_space, local_action_space

    def _get_info(self):
        return super()._get_info()
