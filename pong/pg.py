import numpy as np
import tensorflow as tf

from env import Env
from policy_dqn import PolicyDQN


class PG:
    def __init__(self):
        self.policy = PolicyDQN(input_shape=Env.observations_shape())
        self.memory = []

    def get_action_prob(self, sess, state):
        p = self.policy.predict(sess, state)
        a = 1 if np.random.random() < p else 0
        return a, p

    def get_action(self, sess, state):
        a, _ = self.get_action_prob(sess, state)
        return a

    def get_value(self, sess, state):
        return self.value.predict(sess, state)

    def on_reward(self, s, a, r):
        self.memory.append([s, a, r])

    def clear_memory(self):
        self.memory = []

    def train(self, sess, lr_policy, beta):
        samples = self.memory

        ss, as_, ws = [], [], []
        for [s, a, r] in samples:
            ss.append(s)
            as_.append(a)
            ws.append(r)

        as_ = np.expand_dims(as_, axis=1)
        ws = np.expand_dims(ws, axis=1)

        self.policy.train_on_batch(sess, ss, as_, ws, lr_policy, beta)
