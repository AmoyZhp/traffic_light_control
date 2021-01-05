from collections import deque
import random
from util.type import Transition


class ReplayBuffer(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, transition: Transition):
        """Saves a transition."""
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            return []
        return random.sample(self.buffer, batch_size)

    def get_weight(self):
        weight = {
            "buffer": self.buffer
        }
        return weight

    def set_weight(self, weight):
        self.buffer = weight["buffer"]

    def get_config(self):
        config = {
            "capacity": self.capacity,
        }
        return config

    def __len__(self):
        return len(self.buffer)
