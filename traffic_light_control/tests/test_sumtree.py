import unittest
from hprl.replaybuffer.segment_tree import SegmentTree, SumSegmentTree, MinSegmentTree


class TestSumTree(unittest.TestCase):
    def setUp(self) -> None:
        self.capacity = 4
        self.data_priority = [1, 4, 2, 1]
        self.data = ["a", "b", "c", "d"]

    def test_add(self):
        sum_tree = SumSegmentTree(capacity=self.capacity)
        for i in range(self.capacity):
            sum_tree[i] = self.data_priority[i]
        self.assertEqual(sum_tree.sum(), 8)

    def test_update(self):
        sum_tree = SumSegmentTree(capacity=self.capacity)
        for i in range(self.capacity):
            sum_tree[i] = self.data_priority[i]
        update_value = 3
        sum_tree[0] = update_value
        self.assertEqual(sum_tree.sum(), 10)

    def test_get(self):
        sum_tree = SumSegmentTree(capacity=self.capacity)
        for i in range(self.capacity):
            sum_tree[i] = self.data_priority[i]
        i1 = sum_tree.find_prefixsum_idx(3)
        self.assertEqual(i1, 1)
        i2 = sum_tree.find_prefixsum_idx(6)
        self.assertEqual(i2, 2)
        i3 = sum_tree.find_prefixsum_idx(0)
        self.assertEqual(i3, 0)
        i4 = sum_tree.find_prefixsum_idx(8)
        self.assertEqual(i4, 3)


if __name__ == "__main__":
    unittest.main()