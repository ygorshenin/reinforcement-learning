#!/usr/bin/env python3

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
import argparse
import collections
import math
import numpy as np
import random
import time

from utils import *


MIN_EXPLORATION_RATE = 0.01 * 1 / 200
MAX_EXPLORATION_RATE = 1.0
DISCOUNT = 0.99
MEMORY = 100000

HIDDEN_LAYER = 64
BATCH_SIZE = 32


class DQNAgent:
    def __init__(self, states_dim, actions_dim, exploration_rate, discount, memory):
        model = Sequential()
        model.add(Dense(HIDDEN_LAYER, activation='tanh', input_shape=(states_dim,)))
        model.add(Dense(HIDDEN_LAYER, activation='tanh'))
        model.add(Dense(actions_dim, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop())

        self.model = model
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.exploration_rate = exploration_rate
        self.discount = discount
        self.memory = collections.deque(maxlen=memory)

    def get_action(self, state):
        if np.random.uniform() < self.exploration_rate:
            return np.random.randint(low=0, high=self.actions_dim)
        return np.argmax(self._predict(state)[0])

    def on_reward(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))

        if batch_size == 0:
            return

        memory = random.sample(self.memory, batch_size)

        xs = np.zeros(shape=(batch_size, self.states_dim))
        xs_ = np.zeros(shape=(batch_size, self.states_dim))

        for i, (s, a, r, s_, done) in enumerate(memory):
            xs[i] = s
            xs_[i] = s_

        ys = self._predict(xs)
        ys_ = self._predict(xs_)

        for i, (s, a, r, s_, done) in enumerate(memory):
            target = r
            if not done:
                target += self.discount * np.amax(ys_[i])

            ys[i][a] = target

        self.model.fit(xs, ys, epochs=1, verbose=0)
            
    def _predict(self, state):
        return self.model.predict(state)


def learn_episode(env, agent):
    s = env.reset()

    steps = 0
    while True:
        steps += 1

        a = agent.get_action(s)
        s_, r, done = env.step(a)
        agent.on_reward(s, a, r, s_, done)
        s = s_
        agent.replay(BATCH_SIZE)
        if done:
            break
    return steps


def learn(episodes, model_path):
    exploration_rate = MAX_EXPLORATION_RATE
    exploration_ratio = MIN_EXPLORATION_RATE / MAX_EXPLORATION_RATE

    env = Env()
    agent = DQNAgent(env.states_dim, env.actions_dim, exploration_rate, DISCOUNT, MEMORY)

    try:
        best = 200
        start_time = time.time()
        for episode in range(episodes):
            progress = episode / episodes
            exploration_rate = MAX_EXPLORATION_RATE * math.pow(exploration_ratio, progress)
            agent.exploration_rate = exploration_rate
            steps = learn_episode(env, agent)
            best = min(best, steps)
            time_elapsed = time.time() - start_time
            print('After {}/{} episodes: {}, best: {}, eps: {:.3f}, elapsed: {} secs'.format(episode + 1, episodes, steps, best, exploration_rate, int(time_elapsed)))
            if (episode + 1) % 1000 == 0:
                save_dqn(1000, agent.model, '{}.png'.format(episode + 1))
    except KeyboardInterrupt:
        pass

    print('Saving model to {}...'.format(model_path))
    agent.model.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--episodes', type=int, default=10000,
                        help='number of episodes to train')
    parser.add_argument('--model_path', type=str, default='model.h5y',
                        help='path to save model')
    args = parser.parse_args()
    learn(args.episodes, args.model_path)
