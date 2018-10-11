#!/usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf
import time

from env import Env
from dqn import DQN


EPS = 0.1


def visualize_episode(sess, env, dqn):
    s = env.reset()
    time.sleep(0.02)

    while True:
        qs = dqn.predict(sess, s)
        a = np.argmax(qs)
        if np.random.random() < EPS:
            a = np.random.randint(Env.actions_dim())
        s, _, done = env.step(a)
        time.sleep(0.02)
        if done:
            break


def visualize_episodes(sess, env, dqn):
    while True:
        visualize_episode(sess, env, dqn)


def go(args):
    with tf.Session() as sess:
        dqn = DQN(input_shape=Env.observations_shape(), output_dim=Env.actions_dim())

        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        env = Env(render=True)

        visualize_episodes(sess, env, dqn)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path',
                        type=str,
                        default='model.ckpt',
                        help='Path to save model')
    args = parser.parse_args()
    go(args)
