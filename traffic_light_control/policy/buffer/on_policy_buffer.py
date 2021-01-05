from collections import deque
from policy.core import ReplayBuffer
import random
from util.type import Transition


class OnPolicyBuffer(ReplayBuffer):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action,
              reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            return []
        batch = random.sample(self.buffer, batch_size)
        self.buffer.clear()
        return batch

    def get_weight(self):
        return {}

    def set_weight(self, weight):
        return

    def __len__(self):
        return len(self.buffer)
