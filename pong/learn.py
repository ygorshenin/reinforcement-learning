#!/usr/bin/env python3

from collections import deque
import argparse
import numpy as np
import random
import tensorflow as tf
import time

from dqn import DQN
from env import Env

MEMORY_SIZE = 100000
BATCH_SIZE = 250
REWARD_DECAY = 0.9
EPISODES_TO_TRAIN = 1


def train_on_memory(sess, dqn, memory):
    n = min(len(memory), BATCH_SIZE)
    batch = np.array(random.sample(memory, n))

    ss, as_, ss_, rs = [], [], [], []
    for [s, a, s_, r, _] in batch:
        ss.append(s)
        as_.append(a)
        ss_.append(s_)
        rs.append(r)

    ss = np.array(ss)
    as_ = np.array(as_)
    ss_ = np.array(ss_)
    rs = np.array(rs)

    qs, qs_ = dqn.predict_on_batch(sess, ss), dqn.predict_on_batch(sess, ss_)
    for i in range(n):
        a = as_[i]
        qs[a] = rs[i] + np.max(qs_[i])
    dqn.train_on_batch(sess, ss, qs, lr=0.001)


def train_on_episode(sess, env, dqn, memory):
    reward = 0

    s = env.reset()

    while True:
        qs = dqn.predict(sess, s)
        a = np.argmax(qs)

        s_, r, done = env.step(a)

        memory.append(np.array([s, a, s_, r, done]))
        reward += r

        if r > 1e-9:
            print('Win!')

        if abs(r) > 1e-9:
            print('Training on memory:', r)
            train_on_memory(sess, dqn, memory)

        if done:
            break

    return reward


def train_on_episodes(args):
    with tf.Session() as sess:
        dqn = DQN(input_shape=Env.observations_shape(), output_dim=Env.actions_dim())
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(args.summary, sess.graph)
        saver = tf.train.Saver()

        if args.restore:
            saver.restore(sess, args.model_path)

        env = Env()
        memory = deque(maxlen=MEMORY_SIZE)

        reward, episode = 0, 0

        while True:
            episode += 1
            r = train_on_episode(sess, env, dqn, memory)
            reward += (r - reward) * REWARD_DECAY

            summary = tf.Summary(value=[tf.Summary.Value(tag='running mean', simple_value=reward),
                                        tf.Summary.Value(tag='reward', simple_value=reward)])
            writer.add_summary(summary)

            print('Episode:', episode, 'reward:', reward)
            saver.save(sess, args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path',
                        type=str,
                        default='model.ckpt',
                        help='Path to save model')
    parser.add_argument('--summary',
                        type=str,
                        default='/tmp/pong-v0',
                        help='Path to save summary')
    parser.add_argument('--restore',
                        action='store_true',
                        help='When true model will be pre-loaded')
    args = parser.parse_args()
    train_on_episodes(args)
