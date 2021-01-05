class Policy(object):

    def compute_action(self, state, explore):
        raise NotImplementedError

    def learn_on_batch(self, batch_data):
        raise NotImplementedError


class PolicyWrapper(object):
    def compute_action(self, state):
        raise NotImplementedError

    def record_transition(self, state, action,
                          reward, next_state, done):
        raise NotImplementedError

    def update_policy(self):
        raise NotImplementedError

    def set_mode(self, mode):
        raise NotImplementedError

    def get_mode(self):
        raise NotImplementedError


class ReplayBuffer(object):

    def store(self, state, action,
              reward, next_state, done):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def get_weight(self):
        raise NotImplementedError

    def set_weight(self, weight):
        raise NotImplementedError
