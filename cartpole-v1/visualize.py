#!/usr/bin/env python3

import argparse
import gym
import numpy as np
import time

from utils import *


DELAY_SEC = 0.05


def get_action(q_table, state):
    return np.argmax(q_table[state])


def go(model_path):
    q_table = np.load(model_path)

    env = gym.make('CartPole-v1')
    state = env_reset(env)

    steps = 0
    while True:
        steps += 1

        env.render()
        time.sleep(DELAY_SEC)

        state, _, done = env_step(env, get_action(q_table, state))
        if done:
            break

    print('Passed {} steps'.format(steps))


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparse.add_argument('--model_path', type=str, default='model.npy',
                          help='path to a model to save')
    args = argparse.parse_args()
    go(args.model_path)
