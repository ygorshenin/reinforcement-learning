import collections
import numpy as np
import random
import tensorflow as tf

from env import Env
from policy_dqn import PolicyDQN
from value_dqn import ValueDQN


MEMORY_SIZE = 200000
BATCH_SIZE = 256
EPS = 1e-6

def clamp(value, low, high):
    if value < low:
        value = low
    if value > high:
        value = high
    return value


class AC:
    def __init__(self, discount):
        self.value = ValueDQN(input_shape=Env.observations_shape())
        self.policy = PolicyDQN(input_shape=Env.observations_shape())
        self.memory = collections.deque(maxlen=MEMORY_SIZE)
        self.discount = discount

    def get_action_prob(self, sess, state):
        p = self.policy.predict(sess, state)
        a = 1 if np.random.random() < p else 0
        return a, p

    def get_action(self, sess, state):
        a, _ = self.get_action_prob(sess, state)
        return a

    def get_value(self, sess, state):
        return self.value.predict(sess, state)

    def on_reward(self, s, a, r, s_, done):
        self.memory.append([s, a, r, s_, done])

    def clear_memory(self):
        self.memory.clear()

    def train(self, sess, lr_policy, lr_value):
        # batch_size = min(len(self.memory), BATCH_SIZE)
        # samples = random.sample(self.memory, batch_size)
        samples = self.memory

        ss, as_, ws = [], [], []
        for [s, a, r, s_, done] in samples:
            ss.append(s)
            as_.append(a)
            ws.append(r)

        as_ = np.expand_dims(as_, axis=1)
        ws = np.expand_dims(ws, axis=1)

        self.policy.train_on_batch(sess, ss, as_, ws, lr_policy)
