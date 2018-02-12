import numpy as np
import random
import doctest


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
        self.buffer = np.zeros(2 * self.capacity)
        self.op = op
        
    def set(self, index, value):
        index += self.capacity

        buffer, op = self.buffer, self.op

        buffer[value] = value
        while index > 1:
            parent = index >> 1
            buffer[parent] = op(buffer[index], buffer[index + 1])

    def get(self, index):
        return self.buffer[self.capacity + index]


class SumTree(SegTree):
    def __init__(self, size):
        super().__init__(self, size, lambda x, y: x + y)

    def sample(self):
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

    def probability(self, i):
        return self.buffer[self.capacity + i] / self.buffer[1]


class MaxTree(SegTree):
    def __init__(self, size):
        super().__init__(self, size, max)

    def max(self):
        return self.buffer[1]


class Memory:
    def __init__(self, maxlen):
        self.maxlen = maxlen

        self.items = []
        self.curr = 0

        self.maxs = MaxTree(maxlen)
        self.sums = MaxTree(maxlen)

    def append(self, item, priority):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(item)
            return

        self.items[self.curr] = item
        self.maxs.set(self.curr, priority)
        self.sums.set(self.curr, priority)
        self.curr += 1
        if self.curr >= self.maxlen:
            self.curr = 0

    def sample(self):
        i = self.sums.sample()
        assert i < len(self.items)
        return i

    def get(self, i):
        return self.items[i], self.sums.get(i)

    def set_priority(self, i, priority):
        self.maxs.set(i, priority)
        self.sums.set(i, priority)

    def max_priority(self):
        return self.maxs.max()
