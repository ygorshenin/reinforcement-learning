#!/usr/bin/env python3

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np

from utils import *


LEARNING_RATE = 0.2
DISCOUNT = 1.0
ALPHA = 0.5

MAX_STEPS = 200
MAX_EXPLORATION_RATE = 0.2
MIN_EXPLORATION_RATE = 0.005


class TDLambdaAgent:
    def __init__(self, q_table, learning_rate, discount, exploration_rate):
        self.q_table = q_table
        self.s_table = np.zeros(q_table.shape)
        self.history = []
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate


    def get_action(self, state):
        if np.random.uniform() < self.exploration_rate:
            return np.random.randint(0, 3)
        return np.argmax(self.q_table[state])

    def on_reward(self, curr_state, action, next_state, reward):
        q_table = self.q_table
        s_table = self.s_table

        learning_rate = self.learning_rate
        discount = self.discount

        next_q = q_table[next_state][self.get_action(next_state)]
        delta = reward + discount * next_q - q_table[curr_state][action]

        s_table *= ALPHA * discount
        for a in range(3):
            s_table[curr_state][a] = 0
        s_table[curr_state][action] = 1

        q_table += learning_rate * s_table * delta


def init_q_table():
    n = 1
    for b in BUCKETS:
        n *= b
    return np.zeros(shape=(n, 3))


def learn_episode(env, agent):
    curr_state = env.reset()

    steps = 0
    while True:
        steps += 1

        action = agent.get_action(curr_state)
        next_state, reward, done = env.step(action)
        agent.on_reward(curr_state, action, next_state, reward)
        curr_state = next_state

        if done:
            break

    return steps


def learn(env, q_table, episodes):
    exploration_ratio = MIN_EXPLORATION_RATE / MAX_EXPLORATION_RATE

    success = 0
    steps = []
    for episode in range(episodes):
        progress = episode / episodes
        exploration_rate = MAX_EXPLORATION_RATE * math.pow(exploration_ratio, progress)

        progress = episode / episodes
        agent = TDLambdaAgent(q_table, LEARNING_RATE, DISCOUNT, exploration_rate)
        steps.append(learn_episode(env, agent))

        if steps[-1] < MAX_STEPS:
            if success == 0:
                print('Solved after {} episodes'.format(episode + 1))
            success += 1

        if episode % 100 == 0:
            print('Processed {} episodes'.format(episode))
    print('Success rate: {}/{}'.format(success, episodes))

    plt.plot(steps)
    plt.show()


def go(episodes, model_path):
    env = Env()

    q_table = init_q_table()
    learn(env, q_table, episodes)
    np.save(model_path, q_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--episodes', type=int, default=1000,
                        help='number of episodes to play')
    parser.add_argument('--model_path', type=str, default='model.npy',
                        help='path to save model')
    args = parser.parse_args()
    go(args.episodes, args.model_path)
