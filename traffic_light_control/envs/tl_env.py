import cityflow
from basis.action import Action
from basis.state import State


class TlEnv():
    """encapsulate cityflow by gym api
    """

    def __init__(self, config_path: str, thread_num=1):
        self.eng = cityflow.Engine(config_path, thread_num)
        self.current_phase = 0
        self.next_phase = 1
        self.history = []
        # 应该跟 config 中的 light phase 一致
        self.phase_plan = []

    def step(self, action: Action) -> [State, float, bool, dict]:
        """

        Args:
            action (Action): set traffic light phase

        Returns:
            state (State) : traffic state
            reward (float) : reward
            done (bool) : is terminal or not
            info (dict) : extra information
        """

        last_state = self.__get_state()
        # self.eng.set_tl_phase(action.get_intersection_id(),
        #                       action.get_phase_id())
        self.eng.next_step()
        self.current_phase = action.get_phase_id()
        self.next_phase = (action.get_phase_id() + 1) % 8
        state = self.__get_state()
        reward = 0.0
        done = False
        info = []

        return state, reward, done, info

    def reset(self) -> State:
        self.eng.reset()
        state = State([], 0, 0)
        return state

    def __get_state(self) -> State:
        vehicles_on_lane = self.eng.get_lane_vehicle_count()
        vechile_sum = 0
        for value in vehicles_on_lane.values():
            vechile_sum += value
        print("sum vehicle is ", vechile_sum)
        print("get sum : ", self.eng.get_vehicle_count())
        print("wating vehicles : ", self.eng.get_lane_waiting_vehicle_count())
        state = State(vehicles_on_lane, self.current_phase, self.next_phase)
        return state

    def __get_reward(self) -> float:
        pass

    def __cal_pressure(self, intersection_id: str) -> float:
        pass

    def close(self):
        pass
