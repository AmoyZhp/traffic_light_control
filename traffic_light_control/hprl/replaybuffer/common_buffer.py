from collections import deque
from hprl.util.enum import ReplayBufferTypes
import random
from shutil import Error

from hprl.replaybuffer.core import ReplayBuffer
from hprl.util.typing import Transition


class CommonBuffer(ReplayBuffer):

    def __init__(self, capacity: int):
        self.type = ReplayBufferTypes.Common
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            return []
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()

    def get_config(self):
        config = {
            "type": self.type,
            "capacity": self.capacity
        }
        return config

    def get_weight(self):
        weight = {
            "buffer": self.buffer
        }
        return weight

    def set_weight(self, weight):
        self.buffer = weight.get("buffer")
        if len(self.buffer) > self.capacity:
            raise Error("set weight error, buffer size is bigger than capcity")

    def __len__(self):
        return len(self.buffer)
