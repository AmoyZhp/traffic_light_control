from typing import Dict

from hprl.policy.policy import MultiAgentPolicy, Policy
from hprl.typing import Action, PolicyTypes, SampleBatch, State


class IndependentWrapper(MultiAgentPolicy):
    def __init__(
        self,
        type: PolicyTypes,
        policies: Dict[str, Policy],
    ) -> None:
        self._type = type
        self._policies = policies

    def compute_action(self, state: State) -> Action:
        actions = {}
        for id, p in self._policies.items():
            actions[id] = p.compute_action(state=state.local[id])
        action = Action(local=actions)
        return action

    def learn_on_batch(self, batch_data: Dict[str, SampleBatch]):
        priorities = {}
        for id, p in self._policies.items():
            info = p.learn_on_batch(batch_data[id])
            priorities[id] = info.get("priorities", [])
        return priorities

    def get_weight(self) -> Dict:
        weight = {}
        for id, p in self._policies.items():
            weight[id] = p.get_weight()
        return weight

    def set_weight(self, weight: Dict):
        for id, p in self._policies.items():
            p.set_weight(weight[id])

    def get_config(self) -> Dict:
        config = list(self._policies.values())[0].get_config()
        local_action_space = {}
        local_state_space = {}
        for id, p in self._policies.items():
            local_conf = p.get_config()
            local_action_space[id] = local_conf["action_space"]
            local_state_space[id] = local_conf["state_space"]
        config["local_action_space"] = local_action_space
        config["local_state_space"] = local_state_space
        config["model_id"] = self._type.value
        return config

    def unwrapped(self) -> "MultiAgentPolicy":
        policies = {}
        for id, p in self._policies.items():
            policies[id] = p.unwrapped()
        return IndependentWrapper(type=self._type, policies=policies)
