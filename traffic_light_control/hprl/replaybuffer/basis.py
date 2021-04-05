import logging
import random
from collections import deque
from shutil import Error
from typing import Dict, List, Union

from hprl.replaybuffer.replay_buffer import (MAgentReplayBuffer, ReplayBuffer,
                                             ReplayBufferTypes)
from hprl.util.typing import (MultiAgentSampleBatch, SampleBatch, Transition,
                              TransitionTuple)

logger = logging.getLogger(__name__)


def build(config: Dict, multi: bool):
    capacity = config["capacity"]
    if multi:
        buffer = MAgentBasisBuffer(capacity)
    else:
        buffer = BasisBuffer(capacity)
    return buffer


class BasisBuffer(ReplayBuffer):
    def __init__(self, capacity: int):
        self._type = ReplayBufferTypes.Common
        self._capacity = capacity
        self._buffer = deque(maxlen=capacity)

    @property
    def type(self):
        return self._type

    def store(self, data: TransitionTuple, priorities=None):
        self._buffer.append(data)

    def sample(self, batch_size: int, beta: float = None) -> SampleBatch:
        if batch_size > len(self._buffer):
            return None
        trans = random.sample(self._buffer, batch_size)
        return SampleBatch(transitions=trans)

    def update_priorities(self, idxes: List[int], priorities: List[float]):
        # do nothing
        ...

    def clear(self):
        self._buffer.clear()

    def get_config(self):
        config = {"type": self._type, "capacity": self._capacity}
        return config

    def get_weight(self):
        weight = {"buffer": self._buffer}
        return weight

    def set_weight(self, weight):
        self._buffer = weight.get("buffer")
        if len(self._buffer) > self._capacity:
            raise Error("set weight error, buffer size is bigger than capcity")

    def __len__(self):
        return len(self._buffer)


class MAgentBasisBuffer(MAgentReplayBuffer):
    def __init__(self, capacity: int):
        logger.info("MultiAgentCommonBuffer init")
        self._type = ReplayBufferTypes.Common
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        logger.info("\t capacity : %d", self.capacity)
        logger.info("\t MultiAgentCommonBuffer init done")

    def type(self):
        return self._type

    def store(self, data: Transition, priorities=None):
        self.buffer.append(data)

    def sample(self, batch_size: int, beta=None) -> MultiAgentSampleBatch:
        if batch_size > len(self.buffer):
            return MultiAgentSampleBatch()
        trans = random.sample(self.buffer, batch_size)
        return MultiAgentSampleBatch(transitions=trans)

    def clear(self):
        self.buffer.clear()

    def get_config(self):
        config = {"type": self._type, "capacity": self.capacity}
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
