import numpy as np
import random


def ceil_2pow(n):
    """
    >>> ceil_2pow(10)
    16
    >>> ceil_2pow(0)
    1
    >>> ceil_2pow(32)
    32
    """
    p = 1
    while p < n:
        p *= 2
    return p


class SegTree:
    def __init__(self, size, op):
        self.capacity = ceil_2pow(size)
        self.size = size
        self.buffer = np.zeros(2 * self.capacity)
        self.op = op

    def set(self, index, value):
        index += self.capacity

        buffer, op = self.buffer, self.op

        buffer[index] = value
        while index > 1:
            parent = index >> 1
            buffer[parent] = op(buffer[index], buffer[index ^ 1])
            index = parent

    def get(self, index):
        return self.buffer[self.capacity + index]

    def leaves(self):
        return self.buffer[self.capacity:]


class ProbTree(SegTree):
    def __init__(self, size):
        super().__init__(size, lambda x, y: x + y)

    def sample_1(self):
        assert self.size != 0

        buffer = self.buffer

        curr = 1
        priority = random.uniform(0, buffer[curr])
        while curr < self.capacity:
            assert priority <= buffer[curr]
            left, right = 2 * curr, 2 * curr + 1
            if buffer[left] > priority:
                curr = left
            else:
                curr = right
                priority -= buffer[left]
        assert priority <= buffer[curr]
        return curr - self.capacity

    def sample_n(self, n):
        assert n <= self.size

        samples = []
        weights = []
        for _ in range(n):
            i = self.sample_1()
            samples.append(i)
            weights.append(self.get(i))
            self.set(i, 0)

        for (i, w) in zip(samples, weights):
            self.set(i, w)
        return samples

    def prob(self, i):
        return self.buffer[self.capacity + i] / self.buffer[1]

    def probs(self):
        return [self.prob(i) for i in range(self.size)]


class MaxTree(SegTree):
    def __init__(self, size):
        super().__init__(size, max)

    def max(self):
        return self.buffer[1]
