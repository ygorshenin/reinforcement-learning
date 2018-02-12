import collections
import random

from dqn import DQN
from env import *


# Maximum number of last experience entries.
MEMORY_SIZE = 1000 * 1000
# Number of experience entries used to update model.
BATCH_SIZE = 32
# Frequency of training steps.
STEPS_TO_TRAIN = 4
# Frequency of copying steps.
STEPS_TO_COPY = 10 * 1000

HIDDEN_LAYERS=2
HIDDEN_UNITS=64


class Agent:
    def __init__(self, sess, eps_schedule, lr_schedule):
        self.dqn_online = Agent._make_dqn('online')
        self.dqn_target = Agent._make_dqn('target')

        self.sess = sess
        self.eps_schedule = eps_schedule
        self.lr_schedule = lr_schedule
        self.memory = collections.deque(maxlen=MEMORY_SIZE)

        self.step = 0

    def get_action(self, s):
        if random.random() < self.eps_schedule.get():
            return random.randint(0, ACTIONS_DIM - 1)
        qs = self.dqn_online.predict(np.array([s]), self.sess)
        a = np.argmax(qs)
        return a

    def on_reward(self, s, a, r, s_, done):
        self.memory.append([s, a, r, s_, done])

        self.step += 1
        if self.step % STEPS_TO_TRAIN == 0:
            self._train()
        if self.step % STEPS_TO_COPY == 0:
            self._copy()

    def _train(self):
        n = min(len(self.memory), BATCH_SIZE)

        samples = random.sample(self.memory, n)

        ss, ss_ = [], []
        for [s, a, r, s_, done] in samples:
            ss.append(s)
            ss_.append(s_)

        ss, ss_ = np.array(ss), np.array(ss_)

        qs = self._predict_online(ss)
        qs_ = self._predict_online(ss_)
        ts_ = self._predict_target(ss_)

        for i, [s, a, r, s_, done] in enumerate(samples):
            reward = r

            # There's no need to discount future.
            if not done:
                reward += ts_[i][np.argmax(qs_[i])]
            qs[i][a] = reward

        self.dqn_online.train(ss, qs, self.lr_schedule.get(), self.sess)

    def _copy(self):
        return self.dqn_online.copy_to(self.dqn_target, self.sess)

    def _predict_online(self, ss):
        return self.dqn_online.predict(ss, self.sess)

    def _predict_target(self, ss):
        return self.dqn_target.predict(ss, self.sess)

    @staticmethod
    def _make_dqn(name):
        return DQN(name=name,
                   states_dim=STATES_DIM,
                   actions_dim=ACTIONS_DIM,
                   hidden_layers=HIDDEN_LAYERS,
                   hidden_units=HIDDEN_UNITS)
