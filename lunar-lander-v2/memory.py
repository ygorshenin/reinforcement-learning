from trees import *


class Memory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.size = 0

        self.items = []
        self.curr = 0

        self.ps = ProbTree(maxlen)
        self.ws = MaxTree(maxlen)

    def append(self, item, delta):
        if self.size < self.maxlen:
            self.items.append(item)
            self.size += 1
        else:
            self.items[self.curr] = item

        self.ps.set(self.curr, delta)
        self.ws.set(self.curr, 1.0 / delta)

        self.curr += 1
        if self.curr >= self.maxlen:
            self.curr = 0

    def sample_n(self, n):
        max_w = self.ws.max()

        result = []
        for i in self.ps.sample_n(n):
            item = self.items[i]
            weight = self.ws.get(i) / max_w
            result.append((item, i, weight))
        return result

    def set_delta(self, i, delta):
        self.ps.set(i, delta)
        self.ws.set(i, 1.0 / delta)
