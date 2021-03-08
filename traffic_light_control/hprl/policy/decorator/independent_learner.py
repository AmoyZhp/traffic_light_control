from typing import Dict, List

from hprl.policy.policy import Policy
from hprl.util.typing import Reward, State, Action, Terminal
from hprl.util.typing import Trajectory, Transition


class IndependentLearner(Policy):
    def __init__(
        self,
        agents_id: List[str],
        policies: Dict[str, Policy],
    ) -> None:
        self.agents_id = agents_id
        self.policies = policies

    def compute_action(self, state: State) -> Action:
        local_action = {}
        for id_, policy in self.policies.items():
            local_state_val = state.local.get(id_)
            local_state_wrap = State(central=local_state_val)
            action = policy.compute_action(local_state_wrap)
            local_action[id_] = action.central

        return Action(local=local_action)

    def learn_on_batch(self, batch_data):
        if len(batch_data) == 0 or batch_data is None:
            return
        agents_batch_data: Dict[str, List] = {}
        for id in self.agents_id:
            agents_batch_data[id] = []
        data_slice = batch_data[0]
        if isinstance(data_slice, Transition):
            for data in batch_data:
                for id in self.agents_id:
                    state = data.state.local[id]
                    action = data.action.local[id]
                    reward = data.reward.local[id]
                    next_state = data.next_state.local[id]
                    terminal = data.terminal.local[id]
                    agents_batch_data[id].append(
                        Transition(
                            state=State(central=state),
                            action=Action(central=action),
                            reward=Reward(central=reward),
                            next_state=State(central=next_state),
                            terminal=Terminal(central=terminal),
                        ))
        elif isinstance(data_slice, Trajectory):
            for data in batch_data:
                for id in self.agents_id:
                    states = map(lambda s: State(central=s.local[id]),
                                 data.states)
                    actions = map(lambda a: Action(central=a.local[id]),
                                  data.actions)
                    rewards = map(lambda r: Reward(central=r.local[id]),
                                  data.rewards)
                    agents_batch_data[id].append(
                        Trajectory(
                            states=list(states),
                            actions=list(actions),
                            rewards=list(rewards),
                            terminal=data.terminal,
                        ))
        else:
            raise ("data type error")
        for id in self.agents_id:
            self.policies[id].learn_on_batch(agents_batch_data[id])

    def get_weight(self):
        weight = {}
        for k, v in self.policies.items():
            weight[k] = v.get_weight()
        return weight

    def set_weight(self, weight: Dict):
        for id_, w in weight.items():
            self.policies[id_].set_weight(w)

    def get_config(self):
        # all policy has same config setting
        config = self.policies[self.agents_id[0]].get_config()
        return config

    def unwrapped(self):
        unwrap_policy = {}
        for id, policy in self.policies.items():
            unwrap_policy[id] = policy.unwrapped()
        return IndependentLearner(self.agents_id, unwrap_policy)
