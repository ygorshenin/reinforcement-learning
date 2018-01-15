#!/usr/bin/env python3

import argparse
import numpy as np
import time

from utils import *


DELAY_SEC = 0.05
NUM_EPISODES = 100


def get_action(q_table, state):
    return np.argmax(q_table[state])


def play(env, q_table, visualize):
    state = env.reset()
    steps = 0
    while True:
        steps += 1
        if visualize:
            env.render()
            time.sleep(DELAY_SEC)
        action = get_action(q_table, state)
        state, reward, done = env.step(action)
        if done:
            break
    return steps


def go(model_path, visualize):
    env = Env()
    q_table = np.load(model_path)

    steps = []
    for episode in range(NUM_EPISODES):
        steps.append(play(env, q_table, visualize))
    print('Mean score: {:.2f}'.format(np.mean(steps)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, default='model.npy',
                        help='path to save model')
    parser.add_argument('--visualize', dest='visualize', default=False, action='store_true',
                        help='visualize')
    args = parser.parse_args()
    go(args.model_path, args.visualize)
