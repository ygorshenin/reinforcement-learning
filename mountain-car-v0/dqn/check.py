#!/usr/bin/env python3

from keras.models import load_model
import argparse
import numpy as np
import time

from utils import *


DELAY_SEC = 0.05
NUM_EPISODES = 100


def get_action(model, state):
    return np.argmax(model.predict(state)[0])


def play(env, model, visualize):
    s = env.reset()
    steps = 0
    while True:
        steps += 1
        if visualize:
            env.render()
            time.sleep(DELAY_SEC)
        a = get_action(model, s)
        s, r, done = env.step(a)
        if done:
            break
    return steps


def go(model_path, visualize):
    env = Env()
    model = load_model(model_path)

    steps = []
    for episode in range(NUM_EPISODES):
        steps.append(play(env, model, visualize))
    print('Mean score: {:.2f}'.format(np.mean(steps)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, default='model.h5y',
                        help='path to save model')
    parser.add_argument('--visualize', dest='visualize', default=False, action='store_true',
                        help='visualize')
    args = parser.parse_args()
    go(args.model_path, args.visualize)
