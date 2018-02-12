from trees import *


class Memory:
    def __init__(self, maxlen):
        self.maxlen = maxlen

        self.items = []
        self.curr = 0

        self.ps = ProbTree(maxlen)
        self.ws = MaxTree(maxlen)

    def append(self, item, weight):
        if len(self.buffer) < self.maxlen:
            self.items.append(item)
        else:
            self.items[self.curr] = item

        self.ps.set(self.curr, weight)
        self.ws.set(self.curr, 1.0 / weight)

        self.curr += 1
        if self.curr >= self.maxlen:
            self.curr = 0

    def sample_n(self, n):
        max_w = self.ws.max()

        result = []
        for i in self.ps.sample_n(n):
            item = self.items[i]
            weight = self.ws.get(i) / max_w
            result.append((item, weight))
        return result
