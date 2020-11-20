from collections import deque
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action',  'reward', 'next_state'))


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def sotre(self, transition: Transition):
        """Saves a transition."""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
