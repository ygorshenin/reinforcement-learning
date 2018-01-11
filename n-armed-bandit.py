#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
NUM_LEVERS = 10
NUM_TASKS = 2000
NUM_ROUNDS = 2000

class Bandit:
    def __init__(self):
        self.means = np.random.normal(size=NUM_LEVERS)

    def get_reward(self, action):
        mean = self.means[action]
        return np.random.normal(mean)


class GreedyAgent:
    def __init__(self):
        self.means = np.zeros(NUM_LEVERS)
        self.num_plays = np.zeros(NUM_LEVERS)

    def get_action(self):
        return np.argmax(self.means)

    def on_reward(self, action, reward):
        s = self.means[action] * self.num_plays[action] + reward
        self.num_plays[action] += 1
        self.means[action] = s / self.num_plays[action]


class EpsAgent:
    def __init__(self, eps):
        self.means = np.zeros(NUM_LEVERS)
        self.num_plays = np.zeros(NUM_LEVERS)
        self.eps = eps

    def get_action(self):
        if np.random.uniform() < self.eps:
            return np.random.randint(low=0, high=NUM_LEVERS)
        return np.argmax(self.means)

    def on_reward(self, action, reward):
        s = self.means[action] * self.num_plays[action] + reward
        self.num_plays[action] += 1
        self.means[action] = s / self.num_plays[action]


class SoftmaxAgent:
    def __init__(self, temperature=1.0):
        self.means = np.zeros(NUM_LEVERS)
        self.num_plays = np.zeros(NUM_LEVERS)

        self.num_steps = 0
        self.tmax = 1.0
        self.tmin = 0.01

    def get_action(self):
        self.num_steps += 1

        progress = self.num_steps / NUM_ROUNDS
        temperature = self.tmax * np.power(self.tmin / self.tmax, progress)

        exps = np.exp(self.means / temperature)
        s = np.sum(exps)

        p = np.random.uniform()
        r = 0
        for i in range(NUM_LEVERS):
            r += exps[i] / s
            if p < r:
                return i
        return NUM_LEVERS - 1
    
    def on_reward(self, action, reward):
        s = self.means[action] * self.num_plays[action] + reward
        self.num_plays[action] += 1
        self.means[action] = s / self.num_plays[action]


def play(agent_factory):
    sum_rewards = np.zeros(NUM_ROUNDS)

    for _ in range(NUM_TASKS):
        bandit = Bandit()
        agent = agent_factory()

        reward = 0
        for i in range(NUM_ROUNDS):
            a = agent.get_action()
            r = bandit.get_reward(a)
            agent.on_reward(a, r)
            reward += r
            sum_rewards[i] += reward / (i + 1)

    for i in range(NUM_ROUNDS):
        sum_rewards[i] /= NUM_TASKS

    return sum_rewards


def plot_play(agent_factory, label):
    rewards = play(agent_factory)
    plt.plot(rewards, label=label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N-armed bandit simulator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=SEED, action='store', help='random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    plot_play(lambda: GreedyAgent(), label="greedy")
    plot_play(lambda: EpsAgent(0.1), label="eps 0.1")
    plot_play(lambda: EpsAgent(0.01), label="eps 0.01")

    # for t in [1.0, 0.5, 0.1, 0.01]:
    #     plot_play(lambda: SoftmaxAgent(temperature=t), label='softmax {:2}'.format(t))
    plot_play(lambda: SoftmaxAgent(), label='softmax')

    plt.xlabel('Number of rounds')
    plt.ylabel('Mean reward')
    plt.legend()
    plt.show()
