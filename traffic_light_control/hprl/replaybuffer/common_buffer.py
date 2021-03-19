from collections import deque
import logging
from typing import List, Union
from hprl.replaybuffer.replay_buffer import ReplayBufferTypes
import random
from shutil import Error

from hprl.replaybuffer.replay_buffer import MultiAgentReplayBuffer, ReplayBuffer
from hprl.util.typing import MultiAgentSampleBatch, SampleBatch, Trajectory, Transition, TransitionTuple

logger = logging.getLogger(__package__)


class CommonBuffer(ReplayBuffer):
    def __init__(self, capacity: int):
        self.type = ReplayBufferTypes.Common
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, data: TransitionTuple, priorities=None):
        self.buffer.append(data)

    def sample(self, batch_size: int, beta: float = None) -> SampleBatch:
        if batch_size > len(self.buffer):
            return None
        trans = random.sample(self.buffer, batch_size)
        return SampleBatch(transitions=trans)

    def update_priorities(self, idxes: List[int], priorities: List[float]):
        # do nothing
        ...

    def clear(self):
        self.buffer.clear()

    def get_config(self):
        config = {"type": self.type, "capacity": self.capacity}
        return config

    def get_weight(self):
        weight = {"buffer": self.buffer}
        return weight

    def set_weight(self, weight):
        self.buffer = weight.get("buffer")
        if len(self.buffer) > self.capacity:
            raise Error("set weight error, buffer size is bigger than capcity")

    def __len__(self):
        return len(self.buffer)


class MultiAgentCommonBuffer(MultiAgentReplayBuffer):
    def __init__(self, capacity: int):
        logger.info("MultiAgentCommonBuffer init")
        self.type = ReplayBufferTypes.Common
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        logger.info("\t capacity : %d", self.capacity)
        logger.info("\t MultiAgentCommonBuffer init done")

    def store(self, data: Transition):
        self.buffer.append(data)

    def sample(self, batch_size: int) -> MultiAgentSampleBatch:
        if batch_size > len(self.buffer):
            return MultiAgentSampleBatch()
        trans = random.sample(self.buffer, batch_size)
        return MultiAgentSampleBatch(transitions=trans)

    def clear(self):
        self.buffer.clear()

    def get_config(self):
        config = {"type": self.type, "capacity": self.capacity}
        return config

    def update_priorities(self, idxes: List[int], priorities: List[float]):
        ...

    def get_weight(self):
        weight = {"buffer": self.buffer}
        return weight

    def set_weight(self, weight):
        self.buffer = weight.get("buffer")
        if len(self.buffer) > self.capacity:
            raise Error("set weight error, buffer size is bigger than capcity")

    def __len__(self):
        return len(self.buffer)
