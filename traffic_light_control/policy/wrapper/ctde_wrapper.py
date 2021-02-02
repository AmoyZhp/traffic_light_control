import numpy as np
from policy.core import PolicyWrapper


class CTDEWrapper(PolicyWrapper):

    def __init__(self, policy_, buffer_, local_ids, batch_size, mode="train"):

        self.policy = policy_
        self.buffer = buffer_
        self.local_ids = local_ids
        self.batch_size = batch_size
        self.mode = mode

    def compute_action(self, states):
        explore = True if self.mode == "train" else False
        action = self.policy.compute_action(states, explore)
        return action

    def record_transition(self, states, actions,
                          rewards, next_states, done):
        central_state = np.array(states["central"])
        central_reward = np.array(rewards["central"])
        central_next_state = next_states["central"]

        local_states = {}
        local_rewards = {}
        local_next_states = {}
        local_actions = {}
        for id_ in self.local_ids:
            local_states[id_] = np.array(states["local"][id_])
            local_rewards[id_] = np.array(rewards["local"][id_])
            local_next_states[id_] = np.array(next_states["local"][id_])
            local_actions[id_] = np.array(actions[id_])
        n_state = {
            "central": central_state,
            "local": local_states
        }
        n_rewards = {
            "central": central_reward,
            "local": local_rewards,
        }
        n_actions = local_actions
        n_next_state = {
            "central": central_next_state,
            "local": local_next_states,
        }
        self.buffer.store(n_state, n_actions, n_rewards, n_next_state, done)

    def update_policy(self):
        batch_data = self.buffer.sample(self.batch_size)
        if batch_data is None or len(batch_data) <= 0:
            return {
                "central": 0.0,
                "local": {},
            }
        loss = self.policy.learn_on_batch(batch_data)
        return loss

    def get_weight(self):
        polices_weight = self.policy.get_weight()

        buffer_weight = self.buffer.get_weight()

        weight = {
            "policy": polices_weight,
            "buffer": buffer_weight,
        }
        return weight

    def set_weight(self, weight):
        self.policy.set_weight(weight["policy"])
        if self.mode == "train":
            self.buffer.set_weight(weight["buffer"])

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode
