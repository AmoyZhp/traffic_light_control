import numpy as np

import buffer
import policy
from util.type import Transition


class CentralizedWrapper():

    def __init__(self, config, mode="train"):

        policy_config = config["policy"]
        self.ids = policy_config["local_ids"]
        self.batch_size = policy_config["batch_size"]
        self.policy = policy.get_policy(policy_config["alg_id"], policy_config)

        self.mode = mode
        if self.mode == "train":
            buffer_config = policy_config["buffer"]
            self.buffer = buffer.get_buffer(
                buffer_config["id"],  buffer_config)

    def compute_action(self, states):
        explore = True if self.mode == "train" else False
        action = self.policy.compute_action(states, explore)
        return action

    def record_transition(self, states, actions,
                          rewards, next_states, done):
        trans = self.__process_transition(
            states, actions,
            rewards, next_states, done)
        self.buffer.store(trans)

    def update_policy(self):
        batch_data = self.buffer.sample(self.batch_size)
        central = self.policy.learn_on_batch(batch_data)
        return {"central": central,
                "local": {}}

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

    def __process_transition(self,
                             states, actions,
                             reward, next_states, done):
        processed_s = {}
        processed_a = {}
        processed_ns = {}
        processed_r = np.reshape(reward["central"], (1, 1))
        processed_t = np.array([[0 if done else 1]])
        for id_ in self.ids:
            processed_s[id_] = states[id_]
            processed_a[id_] = np.reshape(actions[id_], (1, 1))
            processed_ns[id_] = next_states[id_]
        return Transition(processed_s, processed_a,
                          processed_r, processed_ns, processed_t)
