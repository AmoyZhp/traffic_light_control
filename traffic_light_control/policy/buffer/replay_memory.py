from collections import deque
import random
from collections import namedtuple
from util.type import Transition


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def sotre(self, transition: Transition):
        """Saves a transition."""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
