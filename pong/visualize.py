#!/usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf
import time

from agent import Agent
from env import Env
from dqn import DQN


DELAY_SEC = 0.02


def visualize_episode(sess, env, agent):
    s = env.reset()
    time.sleep(DELAY_SEC)

    while True:
        a = agent.get_action(sess, s)
        s, _, done = env.step(a)
        time.sleep(DELAY_SEC)
        if done:
            break


def visualize_episodes(sess, env, agent):
    while True:
        visualize_episode(sess, env, agent)


def go(args):
    with tf.Session() as sess:
        dqn = DQN(input_shape=Env.observations_shape())
        agent = Agent(dqn)

        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        env = Env(render=True)

        visualize_episodes(sess, env, agent)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path',
                        type=str,
                        default='model.ckpt',
                        help='Path to save model')
    args = parser.parse_args()
    go(args)
