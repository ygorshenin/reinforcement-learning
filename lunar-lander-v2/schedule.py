import math


class BaseSchedule:
    def __init__(self):
        self.progress = 0

    def on_progress(self, progress):
        self.progress = progress

    def get(self):
        assert False, "Not implemented"


class LogSchedule(BaseSchedule):
    """Log cooling schedule from Simulated Annealing"""
    def __init__(self, maxT, minT):
        BaseSchedule.__init__(self)
        self.maxT = maxT
        self.ratioT = minT / maxT

    def get(self):
        return self.maxT * math.pow(self.ratioT, self.progress)


class ConstSchedule(BaseSchedule):
    """Constant cooling schedule, always returns const"""
    def __init__(self, T):
        BaseSchedule.__init__(self)
        self.T = T

    def get(self):
        return self.T

