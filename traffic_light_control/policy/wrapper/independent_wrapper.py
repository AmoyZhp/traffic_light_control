import numpy as np

from policy.core import PolicyWrapper


class IndependentWrapper(PolicyWrapper):

    def __init__(self, policies, local_ids, batch_size,
                 buffers=None, mode="train"):

        self.policies = policies
        self.local_ids = local_ids
        self.batch_size = batch_size
        self.buffers = buffers
        self.mode = mode

    def compute_action(self, states):
        explore = True if self.mode == "train" else False
        actions = {}
        for id_ in self.local_ids:
            obs = states["local"][id_]
            policy_ = self.policies[id_]
            if obs is None or policy_ is None:
                print("policy id is not exit {}".format(id_))
            action = policy_.compute_action(obs, explore)
            actions[id_] = action
        return actions

    def record_transition(self, states, actions,
                          rewards, next_states, done):
        local_rewards = rewards["local"]
        local_states = states["local"]
        local_next_states = next_states["local"]
        for id_ in self.local_ids:
            print(local_states[id_])
            s = np.array(local_states[id_])
            a = np.array(actions[id_])
            r = np.array(local_rewards[id_])
            terminal = np.array(done)
            ns = np.array(local_next_states[id_])
            buff = self.buffers[id_]
            print(s.shape)
            buff.store(s, a, r, ns, terminal)

    def update_policy(self):
        local_loss = {}
        for id_ in self.local_ids:
            buff = self.buffers[id_]
            batch_data = buff.sample(self.batch_size)
            policy_ = self.policies[id_]
            local_loss[id_] = policy_.learn_on_batch(
                batch_data)
        loss = {
            "central": 0.0,
            "local": local_loss
        }
        return loss

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
        for id_ in self.local_ids:
            policy_weight = weight["policy"][id_]
            self.policies[id_].set_weight(policy_weight)
            if self.mode == "test":
                continue
            self.buffers[id_].set_weight(weight["buffer"][id_])

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode
