#!/usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf
import time

from ac import AC
from env import Env


DELAY_SEC = 0.02


def visualize_episode(sess, env, ac):
    s = env.reset()
    time.sleep(DELAY_SEC)

    while True:
        a, p = ac.get_action_prob(sess, s)
        s, r, done = env.step(a)
        print(a, p, r, ac.get_value(sess, s))
        time.sleep(DELAY_SEC)
        if done:
            break


def visualize_episodes(sess, env, ac):
    while True:
        visualize_episode(sess, env, ac)


def go(args):
    with tf.Session() as sess:
        env = Env(render=True)
        ac = AC(discount=1)

        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        visualize_episodes(sess, env, ac)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path',
                        type=str,
                        default='model.ckpt',
                        help='Path to save model')
    args = parser.parse_args()
    go(args)
