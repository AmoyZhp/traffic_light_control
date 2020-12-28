import numpy as np
import buffer
import policy

from util.type import Transition


class IndependentWrapper():

    def __init__(self, config, mode="train"):

        self.ids = config["local_ids"]

        policy_config = config["policy"]

        self.batch_size = policy_config["batch_size"]

        self.policies = {}
        for id_ in self.ids:
            self.policies[id_] = policy.get_policy(
                policy_config["alg_id"], policy_config)

        self.mode = mode

        if self.mode == "train":
            buffer_config = policy_config["buffer"]
            self.buffers = {}
            for id_ in self.ids:
                self.buffers[id_] = buffer.get_buffer(
                    buffer_config["id"],  buffer_config)

    def compute_action(self, states):
        explore = True if self.mode == "train" else False
        actions = {}
        for id_ in self.ids:
            obs = states[id_]
            policy_ = self.policies[id_]
            if obs is None or policy_ is None:
                print("policy id is not exit {}".format(id_))
            action = policy_.compute_single_action(obs, explore)
            actions[id_] = action
        return actions

    def record_transition(self, states, actions,
                          rewards, next_states, done):
        for id_ in self.ids:
            s = states[id_]
            a = np.reshape(actions[id_], (1, 1))
            r = np.reshape(rewards[id_], (1, 1))
            ns = next_states[id_]
            terminal = np.array([[0 if done else 1]])
            buff = self.buffers[id_]
            buff.store(Transition(s, a, r, ns, terminal))

    def update_policy(self):
        local_loss = {}
        for id_ in self.ids:
            buff = self.buffers[id_]
            batch_data = buff.sample(self.batch_size)
            policy_ = self.policies[id_]
            local_loss[id_] = policy_.learn_on_batch(
                batch_data)
        return local_loss

    def get_weight(self):
        polices_weight = {}
        for id_, p in self.policies.items():
            polices_weight[id_] = p.get_weight()

        buffer_weight = {}
        for id_, b in self.buffers.items():
            buffer_weight[id_] = b.get_weight()

        weight = {
            "policy": polices_weight,
            "buffer": buffer_weight,
        }
        return weight

    def set_weight(self, weight):
        for id_ in self.ids:
            policy_weight = weight["policy"][id_]
            self.policies[id_].set_weight(policy_weight)
            if self.mode == "test":
                continue
            self.buffers[id_].set_weight(weight["buffer"][id_])

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode
