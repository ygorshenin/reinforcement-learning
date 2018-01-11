#!/usr/bin/env python3

import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import time

from utils import *


LEARNING_RATE = 0.1
DISCOUNT = 0.99
MAX_EXPLORATION_RATE = 0.2
MIN_EXPLORATION_RATE = 0.002
MAX_STEPS = 500


class SarsaAgent:
    def __init__(self, q_table, learning_rate, discount, exploration_rate):
        self.q_table = q_table
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate

    def get_action(self, state):
        if np.random.uniform() < self.exploration_rate:
            return np.random.randint(0, 2)
        return np.argmax(self.q_table[state])

    def on_reward(self, curr_state, action, next_state, reward):
        q_table = self.q_table
        learning_rate = self.learning_rate
        discount = self.discount
        next_q = q_table[next_state][self.get_action(next_state)]
        delta = learning_rate * (reward + discount * next_q - q_table[curr_state][action])
        q_table[curr_state][action] += delta


def learn_episode(env, agent):
    curr_state = env_reset(env)

    steps = 0
    while True:
        steps += 1

        action = agent.get_action(curr_state)
        next_state, reward, done = env_step(env, action)

        if done and steps < MAX_STEPS:
            reward = -500
        agent.on_reward(curr_state, action, next_state, reward)
        
        curr_state = next_state

        if done:
            break

    return steps


def learn(env, q_table, episodes):
    exploration_ratio = MIN_EXPLORATION_RATE / MAX_EXPLORATION_RATE

    success = 0
    steps = []
    solved = False
    for episode in range(episodes):
        progress = episode / episodes
        exploration_rate = MAX_EXPLORATION_RATE * math.pow(exploration_ratio, progress)
        agent = SarsaAgent(q_table, LEARNING_RATE, DISCOUNT, exploration_rate)
        steps.append(learn_episode(env, agent))
        if steps[-1] == MAX_STEPS:
            success += 1
            if not solved:
                print('Solved after {} episodes'.format(episode + 1))
                solved = True
    print('Succes rate: {}/{}'.format(success, episodes))
    plt.plot(steps)
    plt.show()


def init_q_table():
    n = 1
    for b in BUCKETS:
        n *= b
    q_table = np.zeros(shape=(n, 2))
    return q_table


def go(episodes, model_path):
    env = gym.make('CartPole-v1')
    q_table = init_q_table()
    learn(env, q_table, episodes)
    np.save(model_path, q_table)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparse.add_argument('--episodes', type=int, default=10000,
                          help='number of episodes to play')
    argparse.add_argument('--model_path', type=str, default='model.npy',
                          help='path to a model to save')
    argparse.add_argument('--seed', type=int, default=42,
                          help='random seed')
    args = argparse.parse_args()

    np.random.seed(args.seed)
    go(args.episodes, args.model_path)
