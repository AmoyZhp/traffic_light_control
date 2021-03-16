import logging
from gym import logger
from hprl.replaybuffer.replay_buffer import SingleAgentReplayBuffer
from typing import List
from hprl.replaybuffer.segment_tree import MinSegmentTree, SumSegmentTree
from hprl.util.typing import BufferData, SampleBatch, TransitionTuple
import random

logger = logging.getLogger(__name__)


class PrioritizedReplayBuffer(SingleAgentReplayBuffer):
    def __init__(self, capacity: int, alpha: float) -> None:

        self.capacity = 1
        while self.capacity < capacity:
            self.capacity = self.capacity << 1
        self._buffer = []
        self._idx = 0
        self._sum_tree = SumSegmentTree(self.capacity)
        self._min_tree = MinSegmentTree(self.capacity)
        self._max_priority = 1.0
        self._alpha = alpha
        logger.info("prioritized replay buffer init")
        logger.info("\t alpha is %f", self._alpha)
        logger.info("\t capacity is %d", self.capacity)

    def store(self, data: TransitionTuple, priority: float = None):
        self._buffer.append(data)
        if priority is None:
            priority = self._max_priority
        self._sum_tree[self._idx] = priority**self._alpha
        self._min_tree[self._idx] = priority**self._alpha
        self._idx = (self._idx + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> SampleBatch:
        if batch_size > len(self._buffer):
            return SampleBatch([], [], [], [])
        idxes = []
        for _ in range(batch_size):
            r = random.random() * self._sum_tree.sum()
            idx = self._sum_tree.find_prefixsum_idx(r)
            idxes.append(idx)

        p_min = self._min_tree.min() / self._sum_tree.sum()
        max_weight = (p_min * len(self._buffer))**(-beta)

        weights = []
        batch_indexes = []
        trans = []
        for idx in idxes:
            p_sample = self._sum_tree[idx] / self._sum_tree.sum()
            weight = (p_sample * len(self._buffer))**(-beta) / max_weight
            weights.append(weight)
            batch_indexes.append(idx)
            trans.append(self._buffer[idx])
        batch = SampleBatch(
            transitions=trans,
            trajectorys=None,
            weights=weights,
            idxes=batch_indexes,
        )
        return batch

    def update_priorities(self, idxes: List[int], priorities: List[float]):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            # plus to prevent from being zero
            self._sum_tree[idx] = priority**self._alpha + 1e-6
            self._min_tree[idx] = priority**self._alpha + 1e-6
            self._max_priority = max(priority, self._max_priority)

    def get_weight(self):
        weight = {
            "buffer": self._buffer,
            "idx": self._idx,
            "max_p": self._max_priority,
            "sum_tree_val": self._sum_tree._value,
            "min_tree_val": self._min_tree._value,
        }
        return weight

    def set_weight(self, weight):
        self._buffer = weight["buffer"]
        self._idx = weight["idx"]
        self._max_priority = weight["max_p"]
        self._sum_tree._value = weight["sum_tree_val"]
        self._min_tree._value = weight["min_tree_val"]

    def get_config(self):
        config = {
            "capacity": self.capacity,
            "alpha": self._alpha,
        }
        return config

    def clear(self):
        self.__init__(self.capacity, self._alpha)