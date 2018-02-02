#!/usr/bin/env python3

import argparse
import collections
import logging
import numpy as np
import random
import sys
import tensorflow as tf

from env import *
from model import Model


# All episodes are finite, there is no need in discount
DISCOUNT = 1.0

LEARNING_RATE = 0.00025

# Maximum number of last experience entries.
MEMORY_SIZE = 1000 * 1000

# Number of experience entries used to update model.
BATCH_SIZE = 32

EPS = 0.1

EPISODES_TO_WIN = 100
MEAN_REWARD_TO_WIN = 200

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Agent:
    def __init__(self, session):
        self.model = Model()
        self.session = session
        self.memory = collections.deque(maxlen=MEMORY_SIZE)

    def get_action(self, s):
        if random.random() < EPS:
            return random.randint(0, ACTIONS_DIM - 1)
        qs = self.model.predict(np.array([s]), self.session)
        a = np.argmax(qs)
        return a

    def on_reward(self, s, a, r, s_, done):
        self.memory.append([s, a, r, s_, done])

    def train(self):
        n = min(len(self.memory), BATCH_SIZE)
        sample = random.sample(self.memory, n)

        ss, ss_ = [], []
        for [s, a, r, s_, done] in sample:
            ss.append(s)
            ss_.append(s_)
        ss, ss_ = np.array(ss), np.array(ss_)
        qs, qs_ = self._predict(ss), self._predict(ss_)

        for i, [s, a, r, s_, done] in enumerate(sample):
            reward = r
            if not done:
                reward += DISCOUNT * np.amax(qs_[i])
            qs[i][a] = reward
        self._train(ss, qs, LEARNING_RATE)

    def _predict(self, s):
        return self.model.predict(s, self.session)

    def _train(self, ss, qs, lr):
        return self.model.train(ss, qs, lr, self.session)


def learn_episode(agent, env):
    s = env.reset()

    reward = 0
    steps = 0

    while True:
        steps += 1

        a = agent.get_action(s)
        s_, r, done  = env.step(agent.get_action(s))

        agent.on_reward(s, a, r, s_, done)
        agent.train()

        reward += r
        s = s_

        if done:
            break

    return steps, reward

def learn(args):
    with tf.Session() as session:

        agent = Agent(session)
        env = Env(render=args.render)

        session.run(tf.global_variables_initializer())

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
