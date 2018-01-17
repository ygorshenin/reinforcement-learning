#!/usr/bin/env python3

import argparse
import collections
import math
import numpy as np
import random
import tensorflow as tf
import time

from utils import *


MIN_EXPLORATION_RATE = 0.01 * 1 / 200
MAX_EXPLORATION_RATE = 1.0
DISCOUNT = 1.0

HIDDEN_LAYER = 64

LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.997
EPISODES_SKIP = 100

MEMORY_SIZE = 100000
BATCH_SIZE = 32

STEPS_TO_WIN = 100


class DQNAgent:
    def __init__(self, states_dim, actions_dim, exploration_rate, discount):
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.exploration_rate = exploration_rate
        self.discount = discount
        self.memory = collections.deque(maxlen=MEMORY_SIZE)

        self.s = tf.placeholder(tf.float32,
                                shape=[None, states_dim],
                                name='s')
        self.learning_rate = tf.placeholder(tf.float32,
                                            shape=None,
                                            name='learning_rate')

        H1 = tf.layers.dense(inputs=self.s,
                             units=HIDDEN_LAYER,
                             activation=tf.tanh,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        H2 = tf.layers.dense(inputs=self.s,
                             units=HIDDEN_LAYER,
                             activation=tf.tanh,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())        
        self.q_pred = tf.layers.dense(H2,
                                      units=actions_dim,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.q_actual = tf.placeholder(tf.float32,
                                       shape=[None, actions_dim],
                                       name='q_actual')

        loss = tf.losses.mean_squared_error(self.q_pred, self.q_actual)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(loss)

    def get_action(self, s, sess):
        if np.random.uniform() < self.exploration_rate:
            return np.random.randint(low=0, high=self.actions_dim)
        return np.argmax(sess.run(self.q_pred, feed_dict={self.s: s}))

    def on_reward(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def train(self, learning_rate, sess):
        batch_size = min(len(self.memory), BATCH_SIZE)
        samples = random.sample(self.memory, batch_size)

        ss = np.zeros(shape=(batch_size, self.states_dim))
        ss_ = np.zeros(shape=(batch_size, self.states_dim))
        for i, (s, a, r, s_, done) in enumerate(samples):
            ss[i] = s
            ss_[i] = s_

        qs = self.predict(ss, sess)
        qs_ = self.predict(ss_, sess)

        for i, (s, a, r, s_, done) in enumerate(samples):
            q = r
            if not done:
                q += self.discount * np.amax(qs_[i])
            qs[i][a] = q
        sess.run(self.train_op,
                 feed_dict={self.s: ss, self.q_actual: qs, self.learning_rate: learning_rate})
        
    def predict(self, s, sess):
        return sess.run(self.q_pred, feed_dict={self.s: s})


class Model:
    def __init__(self, agent, sess):
        self.agent = agent
        self.sess = sess

    def predict(self, s):
        return self.agent.predict(s, self.sess)


def learn_episode(env, agent, learning_rate, sess):
    s = env.reset()

    steps = 0
    while True:
        steps += 1

        a = agent.get_action(s, sess)
        s_, r, done = env.step(a)
        agent.on_reward(s, a, r, s_, done)
        agent.train(learning_rate, sess)
        s = s_
        if done:
            break
    return steps


def learn(episodes, model_path, sess):
    exploration_rate = MAX_EXPLORATION_RATE
    exploration_ratio = MIN_EXPLORATION_RATE / MAX_EXPLORATION_RATE

    env = Env()
    agent = DQNAgent(env.states_dim, env.actions_dim, exploration_rate, DISCOUNT)
    sess.run(tf.global_variables_initializer())

    best = 200
    start_time = time.time()
    scores = []
    for episode in range(episodes):
        progress = episode / episodes
        agent.exploration_rate = exploration_rate
        exploration_rate = MAX_EXPLORATION_RATE * math.pow(exploration_ratio, progress)
        learning_rate = LEARNING_RATE * math.pow(LEARNING_RATE_DECAY, episode / EPISODES_SKIP)
        steps = learn_episode(env, agent, learning_rate, sess)
        scores.append(steps)
        time_elapsed = time.time() - start_time
        print('After {}/{} episodes: {}, mean: {}, eps: {:.3f}, lr: {:.5f}, elapsed: {} secs'.format(episode + 1,episodes, steps, np.mean(scores[-STEPS_TO_WIN:]), exploration_rate, learning_rate, int(time_elapsed)))
        if (episode + 1) % 1000 == 0:
            save_dqn(1000, Model(agent, sess), 'after-{}-episodes.png'.format(episode + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--episodes', type=int, default=10000,
                        help='number of episodes to train')
    parser.add_argument('--model_path', type=str, default='model.h5y',
                        help='path to save model')
    args = parser.parse_args()

    with tf.Session() as sess:
        learn(args.episodes, args.model_path, sess)
