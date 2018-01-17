import collections
import numpy as np
import random

from policy import Policy
from value import Value


# Number of units in a single hidden layer.
HIDDEN_UNITS = 64

# Maximum number of last state transitions.
MEMORY_SIZE = 1000000

# Number of items from a history used to train policy and value.
BATCH_SIZE = 32


class ActorCritic:
    def __init__(self, env, discount):
        self.discount = discount
        self.memory = collections.deque(maxlen=MEMORY_SIZE)
        self.policy = Policy(env, HIDDEN_UNITS)
        self.value = Value(env, HIDDEN_UNITS)

    def get_action(self, s, sess):
        return self.policy.get_action(s, sess)

    def get_value(self, s, sess):
        return self.value.get_value(s, sess)

    def on_reward(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def train(self, lr_policy, lr_value, sess):
        batch_size = min(len(self.memory), BATCH_SIZE)
        samples = random.sample(self.memory, batch_size)

        ss = np.zeros(shape=[batch_size, 2])
        ss_ = np.zeros(shape=[batch_size, 2])
        acts = np.zeros(shape=[batch_size, 1])
        for i, (s, a, r, s_, done) in enumerate(samples):
            ss[i] = s
            ss_[i] = s_
            acts[i] = a

        pvs = self.value.get_value(ss, sess)
        vs_ = self.value.get_value(ss_, sess)

        vs = np.zeros(shape=[batch_size, 1])
        advantages = np.zeros(shape=[batch_size, 1])
        for i, (s, a, r, s_, done) in enumerate(samples):
            vs[i] = r
            if not done:
                vs[i] += self.discount * vs_[i]
        advantages = vs - pvs

        self.value.train(ss, vs, lr_value, sess)
        self.policy.train(ss, acts, advantages, lr_policy, sess)
