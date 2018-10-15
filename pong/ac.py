import collections
import numpy as np
import random

from env import Env
from policy_dqn import PolicyDQN
from value_dqn import ValueDQN


MEMORY_SIZE = 100000
BATCH_SIZE = 128
EPS = 1e-6

def clamp(value, low, high):
    if value < low:
        value = low
    if value > high:
        value = high
    return value


class AC:
    def __init__(self):
        self.value = ValueDQN(input_shape=Env.observations_shape())
        self.policy = PolicyDQN(input_shape=Env.observations_shape())
        self.memory = collections.deque(maxlen=MEMORY_SIZE)

    def get_action_prob(self, sess, state):
        p = self.policy.predict(sess, state)
        a = 1 if np.random.random() < p else 0
        return a, p

    def get_value(self, sess, state):
        return self.value.predict(sess, state)

    def on_reward(self, s, a, p, r, s_, done):
        self.memory.append([s, a, p, r, s_, done])

    def train(self, sess, lr_policy, lr_value):
        batch_size = min(len(self.memory), BATCH_SIZE)
        samples = random.sample(self.memory, batch_size)

        ss, ss_ = [], []
        for [s, a, p, r, s_, done] in samples:
            ss.append(s)
            ss_.append(s_)

        values_pred = self.value.predict_on_batch(sess, ss)
        values_pred_ = self.value.predict_on_batch(sess, ss_)

        values, weights = np.zeros(shape=[batch_size, 1]), np.zeros(shape=[batch_size, 1])
        for i, [s, a, p, r, _s, done] in enumerate(samples):
            reward = r
            if not done:
                reward += values_pred_[i]
            values[i][0] = reward
            if a == 1:
                weights[i][0] = (reward - values_pred[i]) / clamp(p, EPS, 1 - EPS)
            else:
                weights[i][0] = (values_pred[i] - reward) / clamp(1 - p, EPS, 1 - EPS)
        self.policy.train_on_batch(sess, ss, weights, lr_policy)
        self.value.train_on_batch(sess, ss, values, lr_value)
