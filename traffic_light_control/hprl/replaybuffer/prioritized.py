import logging
import operator
import random
from typing import Dict, List

from gym import logger
from hprl.replaybuffer.replay_buffer import MAgentReplayBuffer, ReplayBuffer
from hprl.typing import (MultiAgentSampleBatch, ReplayBufferTypes, SampleBatch,
                         Transition, TransitionTuple)

logger = logging.getLogger(__name__)


def build(config: Dict, mulit: bool):
    alpha = config["alpha"]
    capacity = config["capacity"]
    if mulit:
        buffer = MAgentPER(capacity=capacity, alpha=alpha)
    else:
        buffer = PrioritizedBuffer(capacity=capacity, alpha=alpha)
    return buffer


class PrioritizedBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float) -> None:
        self._type = ReplayBufferTypes.Prioritized
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

    def type(self):
        return self._type

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
        while len(idxes) != batch_size:
            r = random.random() * self._sum_tree.sum()
            idx = self._sum_tree.find_prefixsum_idx(r)
            if idx in idxes:
                continue
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
            "type": self._type,
            "capacity": self.capacity,
            "alpha": self._alpha,
        }
        return config

    def clear(self):
        self.__init__(self.capacity, self._alpha)


class MAgentPER(MAgentReplayBuffer):
    def __init__(self, capacity: int, alpha: float) -> None:
        self._type = ReplayBufferTypes.Prioritized
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

    def type(self):
        return self._type

    def store(self, data: Transition, priority: float = None):
        self._buffer.append(data)
        if priority is None:
            priority = self._max_priority
        self._sum_tree[self._idx] = priority**self._alpha
        self._min_tree[self._idx] = priority**self._alpha
        self._idx = (self._idx + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> MultiAgentSampleBatch:
        if batch_size > len(self._buffer):
            return MultiAgentSampleBatch()
        idxes = []
        while len(idxes) != batch_size:
            r = random.random() * self._sum_tree.sum()
            idx = self._sum_tree.find_prefixsum_idx(r)
            if idx in idxes:
                continue
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
        batch = MultiAgentSampleBatch(
            transitions=trans,
            weigths=weights,
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


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (
            capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1,
                                           node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1,
                                        node_end))

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx],
                                               self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity=capacity,
                                             operation=operator.add,
                                             neutral_element=0.0)

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity=capacity,
                                             operation=min,
                                             neutral_element=float('inf'))

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class SumTree():
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = [0 for _ in range(self.capacity * 2)]
        self.data_buffer = [None for _ in range(self.capacity)]
        self.data_index = 0
        self.data_num = 0

    def add(self, data, priority: float):
        self.data_buffer[self.data_index] = data
        self.update(self.data_index, priority)
        self.data_index = (self.data_index + 1) % self.capacity
        self.data_num = max(self.data_num + 1, self.capacity)

    def update(self, data_index, priority: float):
        tree_index = data_index + self.capacity - 1
        change = priority - self.tree[tree_index]
        self._backward(tree_index, change)

    def sum(self):
        return self.tree[0]

    def get(self, num: float):
        tree_index = self._walk(0, num)
        data_index = tree_index - self.capacity + 1
        return data_index

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity
        return self.data_buffer[id]

    def _backward(self, index: int, change: float):
        self.tree[index] += change
        if index <= 0:
            return
        parent = (index - 1) // 2
        self._backward(parent, change)

    def _walk(self, index: int, num: float):

        left_index = index * 2 + 1
        right_index = index * 2 + 2
        if (left_index >= self.capacity * 2
                or right_index >= self.capacity * 2):
            return index
        left_v = self.tree[left_index]
        if num <= left_v:
            return self._walk(left_index, num)
        else:
            return self._walk(right_index, num - left_v)
