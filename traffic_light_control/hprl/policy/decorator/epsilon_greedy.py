import logging
from typing import Dict, List

import numpy as np

from hprl.util.typing import Action, SampleBatch, State, Transition
from hprl.policy.policy import MultiAgentPolicy, Policy

logger = logging.getLogger(__package__)


class EpsilonGreedy(Policy):
    """
        this class is a wrapper
        usually to wrapping the q-learning like policy.
    """
    def __init__(
        self,
        inner_policy: Policy,
        eps_frame: int,
        eps_init: float,
        eps_min: float,
        action_space,
    ) -> None:
        self.inner_policy = inner_policy
        self.step = 0
        self.eps_frame = eps_frame
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps = eps_init
        self.action_space = action_space

    def compute_action(self, state: np.ndarray) -> int:
        self.step = min(self.step + 1, self.eps_frame)
        self.eps = max(self.eps_init - self.step / self.eps_frame,
                       self.eps_min)
        if np.random.rand() < self.eps:
            action = np.random.choice(range(self.action_space), 1).item()
            return action
        return self.inner_policy.compute_action(state)

    def learn_on_batch(self, sample_batch: SampleBatch):
        return self.inner_policy.learn_on_batch(sample_batch)

    def get_weight(self):
        weight = self.inner_policy.get_weight()
        weight["step"] = self.step
        return weight

    def set_weight(self, weight: Dict):
        self.step = weight.get("step")
        return self.inner_policy.set_weight(weight)

    def get_config(self):
        config = {
            "eps_frame": self.eps_frame,
            "eps_init": self.eps_init,
            "eps_min": self.eps_min
        }
        config.update(self.inner_policy.get_config())
        return config

    def unwrapped(self):
        return self.inner_policy


class MultiAgentEpsilonGreedy(MultiAgentPolicy):
    """
        this class is a wrapper
        usually to wrapping the q-learning like policy.
    """
    def __init__(
        self,
        agents_id: List[str],
        inner_policy: MultiAgentPolicy,
        eps_frame: int,
        eps_init: float,
        eps_min: float,
        action_space,
    ) -> None:
        logger.info("Multi Agent Epsilon Greedy init")

        self.agents_id = agents_id
        self.inner_policy = inner_policy
        self.step = 0
        self.eps_frame = eps_frame
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps = eps_init
        self.action_space = action_space

        logger.info("\t eps frame : %d", self.eps_frame)
        logger.info("\t eps min : %f", self.eps_min)
        logger.info("\t eps init : %f", self.eps_init)
        logger.info("Multi Agent Epsilon Greedy init done")

    def compute_action(self, state: State) -> Action:
        self.step = min(self.step + 1, self.eps_frame)
        self.eps = max(self.eps_init - self.step / self.eps_frame,
                       self.eps_min)
        if np.random.rand() < self.eps:
            actions = {}
            for id in self.agents_id:
                actions[id] = np.random.choice(
                    range(self.action_space[id]),
                    1,
                ).item()
            return Action(local=actions)
        return self.inner_policy.compute_action(state)

    def learn_on_batch(self, batch_data: List[Transition]):
        return self.inner_policy.learn_on_batch(batch_data)

    def get_weight(self):
        weight = self.inner_policy.get_weight()
        weight["step"] = self.step
        return weight

    def set_weight(self, weight: Dict):
        self.step = weight.get("step")
        return self.inner_policy.set_weight(weight)

    def get_config(self):
        config = {
            "eps_frame": self.eps_frame,
            "eps_init": self.eps_init,
            "eps_min": self.eps_min
        }
        config.update(self.inner_policy.get_config())
        return config

    def unwrapped(self):
        return self.inner_policy
