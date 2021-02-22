from collections import deque
import random

from hprl.replaybuffer.core import ReplayBuffer
from hprl.util.typing import Transition


class CommonBuffer(ReplayBuffer):

    def __init__(self, capacity: int):
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

    def get_weight(self):
        ...

    def set_weight(self, weight):
        ...

    def __len__(self):
        return len(self.buffer)
