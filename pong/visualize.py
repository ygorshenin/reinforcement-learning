#!/usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf
import time

from env import Env
from pg import PG


def visualize_episode(sess, env, pg):
    s = env.reset()

    while True:
        a = pg.get_action(sess, s)
        s, _, done = env.step(a)
        if done:
            break


def visualize_episodes(sess, env, pg):
    while True:
        visualize_episode(sess, env, pg)


def go(args):
    with tf.Session() as sess:
        env = Env(render=True)
        pg = PG()

        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        visualize_episodes(sess, env, pg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path',
                        type=str,
                        default='model.ckpt',
                        help='Path to save model')
    args = parser.parse_args()
    go(args)
