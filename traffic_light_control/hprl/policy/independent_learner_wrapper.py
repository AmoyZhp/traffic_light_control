from typing import Dict, List

from hprl.util.typing import Reward, State, Action, Transition
from hprl.policy.core import Policy


class ILearnerWrapper(Policy):
    def __init__(self,
                 agents_id: List[str],
                 policies: Dict[str, Policy]
                 ) -> None:
        super().__init__()
        self.agents_id = agents_id
        self.policies = policies

    def compute_action(self, state: State) -> Action:
        local_action = {}
        for id_, policy in self.policies.items():
            local_state_val = state.local.get(id_)
            local_state_wrap = State(
                central=local_state_val,
            )
            action = policy.compute_action(
                local_state_wrap
            )
            local_action[id_] = action

        return Action(local=local_action)

    def learn_on_batch(self, batch_data: List[Transition]):
        if len(batch_data) == 0 or batch_data == None:
            return
        agents_batch_data: Dict[str, List[Transition]] = {}
        for id_ in self.agents_id:
            agents_batch_data[id_] = []

        for data in batch_data:
            for id_ in self.agents_id:
                state = data.state.local[id_]
                action = data.action.local[id_]
                reward = data.reward.local[id_]
                next_state = data.next_state.local[id_]
                agents_batch_data[id_].append(
                    Transition(
                        state=State(central=state),
                        action=Action(central=action),
                        reward=Reward(central=reward),
                        next_state=State(central=next_state),
                        terminal=data.terminal
                    )
                )
        for id in self.agents_id:
            self.policies[id].learn_on_batch(
                agents_batch_data[id_]
            )

    def get_weight(self):
        ...

    def set_weight(self, weight):
        ...

    def unwrapped(self):
        return self.policies
