from basis.action import Action
from basis.state import State
from policy.dqn import DQNConfig, DQN
from policy.buffer.replay_memory import ReplayMemory, Transition
from policy.net.single_intersection import SingleIntesection


class DQNAgent():
    def __init__(self, intersection_id: str, config: DQNConfig) -> None:
        self.intersection_id = intersection_id
        self.policy = DQN(memory=ReplayMemory(config.capacity),
                          target_net=SingleIntesection(
                              config.state_space, config.action_space),
                          acting_net=SingleIntesection(
                              config.state_space, config.action_space),
                          config=config)

    def act(self, state: State) -> Action:
        act = self.policy.select_action(state.to_tensor())
        keep = True
        if act == 0:
            keep = False
        return Action(keep_phase=keep, intersection_id=self.intersection_id)

    def eval_act(self, state: State) -> Action:
        act = self.policy.select_eval_action(state.to_tensor())
        keep = True
        if act == 0:
            keep = False
        return Action(keep_phase=keep, intersection_id=self.intersection_id)

    def store(self, transition: Transition):
        self.policy.store_transition(transition)

    def update_policy(self):
        return self.policy.update()
