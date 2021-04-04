import logging
from typing import Dict, List, Tuple
import hprl
import numpy as np
from envs.cityflow.road import Road, RoadLink
from envs.cityflow.intersections import Intersection

logger = logging.getLogger(__name__)


class CityFlow(hprl.MultiAgentEnv):
    def __init__(
        self,
        eng,
        id: str,
        max_time: int,
        interval: int,
        intersections: Dict[str, Intersection],
    ) -> None:
        self._type = "CityFlow"
        self._eng = eng
        self._id = id
        self._intersections = intersections
        self._intersections_id = list(self._intersections.keys())
        self._interval = interval
        self._max_time = max_time
        self._time = 0

        self._roads: Dict[str, Road] = {}
        for inter in self._intersections.values():
            roads = inter.roads
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
        self._time = 0
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

        # lane_vehicles is vehicle ids on each lane.
        # lane id as key and list of vehicle id as value.
        lane_vehicles: Dict[str, List[str]] = self._eng.get_lane_vehicles()

        # vehicle_speed is speed of each vehicle
        # vehicle id as key and corresponding speed as value.
        vehicle_speed: Dict[str, float] = self._eng.get_vehicle_speed()

        # the outer Dict key is lane id, value is a vehicles:speed dict
        # the inner Dict key is vehicles id, value is corresonding speed
        vehicles_speed_pool: Dict[str, Dict[str, float]] = {}
        for land_id, vehicle_ids in lane_vehicles.items():
            vehicles_speed_pool[land_id] = {}
            for v_id in vehicle_ids:
                vehicles_speed_pool[land_id][v_id] = vehicle_speed[v_id]

        for inter in self._intersections.values():
            inter.update(
                vehicles_data_pool=vehicles_dict,
                waiting_vehicles_data_pool=waiting_vehicles_dict,
                vehicles_speed_pool=vehicles_speed_pool,
            )

    def _compute_state(self) -> hprl.State:
        local_state = {}
        for id, inter in self._intersections.items():
            local_state[id] = inter.tensor

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
            local_reward[id] = -inter.pressure

        central_reward = self._compute_avg_speed_rate()

        reward = hprl.Reward(central=central_reward, local=local_reward)
        return reward

    def _compute_avg_speed_rate(self):
        avg_speed_sum = 0.0
        vehicles_count = 0
        for road in self._roads.values():
            avg_speed_sum += road.avg_speed_sum
            vehicles_count += road.vehicles
        avg_speed_rate = avg_speed_sum / vehicles_count

        return avg_speed_rate

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
            local_state_space[id] = iter.state_space
            local_action_space[id] = len(iter._phases_plan)
        return local_state_space, local_action_space

    def _log_init_info(self):
        logger.info("City Flow Env init info : ")
        logger.info("env id : %s", self._id)
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
            "type": self._type,
            "max_time": self._max_time,
            "interval": self._interval,
            "id": self._id,
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
    def id(self):
        return self._id


class MaxPressure(CityFlow):
    def __init__(
        self,
        eng,
        id: str,
        max_time: int,
        interval: int,
        intersections: Dict[str, Intersection],
    ) -> None:
        super().__init__(eng, id, max_time, interval, intersections)

    def _compute_state(self) -> hprl.State:
        local_state = {}
        for id in self._intersections_id:
            item = self._intersections[id]
            phase_plans: List[List[int]] = item._phases_plan
            roadlinks: List[RoadLink] = item._roadlinks
            phases_pressure = []
            for phase in phase_plans:
                pressure = 0.0
                for rlink_index in phase:
                    rlink = roadlinks[rlink_index]
                    pressure += rlink.pressure
                phases_pressure.append(pressure)
            local_state[id] = phases_pressure
        state = hprl.State(local=local_state)
        return state

    def _log_init_info(self):
        ...
