import numpy as np

from env import Env


class Agent:
    def __init__(self, dqn, eps=0.1):
        self.dqn = dqn
        self.eps = eps

    def get_action(self, sess, s):
        if np.random.random() < self.eps:
            return np.random.randint(low=0, high=Env.actions_dim())
        qs = self.dqn.predict(sess, s)
        return np.argmax(qs)
