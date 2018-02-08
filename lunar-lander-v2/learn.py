#!/usr/bin/env python3

import argparse
import collections
import logging
import numpy as np
import random
import sys
import tensorflow as tf

from env import *
from dqn import DQN


# All episodes are finite, there is no need in discount
DISCOUNT = 1.0

LEARNING_RATE = 0.0001

# Maximum number of last experience entries.
MEMORY_SIZE = 1000 * 1000

# Number of experience entries used to update model.
BATCH_SIZE = 32

EPS = 0.1

HIDDEN_LAYERS=2
HIDDEN_UNITS=64

EPISODES_TO_WIN = 100
MEAN_REWARD_TO_WIN = 200

STEPS_TO_TRAIN = 4
STEPS_TO_COPY = 10 * 1000

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Agent:
    def __init__(self, sess):
        self.dqn_online = Agent._make_dqn('online')
        self.dqn_target = Agent._make_dqn('target')

        self.sess = sess
        self.memory = collections.deque(maxlen=MEMORY_SIZE)

        self.step = 0

    def get_action(self, s):
        if random.random() < EPS:
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
        sample = random.sample(self.memory, n)

        ss, ss_ = [], []
        for [s, a, r, s_, done] in sample:
            ss.append(s)
            ss_.append(s_)
        ss, ss_ = np.array(ss), np.array(ss_)

        qs = self._predict_online(ss)
        qs_ = self._predict_online(ss_)
        ts_ = self._predict_target(ss_)

        for i, [s, a, r, s_, done] in enumerate(sample):
            reward = r
            if not done:
                reward += DISCOUNT * ts_[i][np.argmax(qs_[i])]
            qs[i][a] = reward
        self.dqn_online.train(ss, qs, LEARNING_RATE, self.sess)

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


def learn_episode(agent, env):
    s = env.reset()

    reward = 0
    steps = 0

    while True:
        steps += 1

        a = agent.get_action(s)
        s_, r, done  = env.step(agent.get_action(s))

        agent.on_reward(s, a, r, s_, done)
        reward += r
        s = s_

        if done:
            break

    return steps, reward

def learn(args):
    with tf.Session() as sess:

        agent = Agent(sess)
        env = Env(render=args.render)

        sess.run(tf.global_variables_initializer())

        rewards = []
        for episode in range(args.episodes):
            logger.info('Learning episode {}...'.format(episode))
            steps, reward = learn_episode(agent, env)
            logger.info('Steps: {}, reward: {}'.format(steps, reward))
            rewards.append(reward)
            logger.info('Running mean: {:.2f}'.format(np.mean(rewards[-EPISODES_TO_WIN:])))


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparse.add_argument('--episodes', type=int, default=1000,
                          help='number of episodes')
    argparse.add_argument('--render', default=False, action='store_true',
                          help='render learning process')
    args = argparse.parse_args()
    learn(args)
