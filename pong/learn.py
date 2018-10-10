#!/usr/bin/env python3

from collections import deque
import numpy as np
import random
import time

from dqn import DQN
from env import Env

MEMORY_SIZE = 100000
BATCH_SIZE = 250
REWARD_DECAY = 0.9
EPISODES_TO_TRAIN = 1


def train_on_memory(dqn, memory):
    n = min(len(memory), BATCH_SIZE)
    batch = np.array(random.sample(memory, n))

    ss, as_, ss_, rs = [], [], [], []
    for [s, a, s_, r, _] in batch:
        ss.append(s)
        as_.append(a)
        ss_.append(s_)
        rs.append(r)

    ss = np.array(ss)
    as_ = np.array(as_)
    ss_ = np.array(ss_)
    rs = np.array(rs)

    qs, qs_ = dqn.predict(ss), dqn.predict(ss_)
    for i in range(n):
        a = as_[i]
        qs[a] = rs[i] + np.max(qs_[i])
    dqn.train(ss, qs)


def train_on_episode(env, dqn, memory):
    reward = 0

    s = env.reset()

    while True:
        qs = dqn.predict(np.array([s]))[0]
        a = np.argmax(qs)

        s_, r, done = env.step(a)

        memory.append(np.array([s, a, s_, r, done]))
        reward += r

        if r > 1e-9:
            print('Win!')

        if abs(r) > 1e-9:
            print('Training on memory:', r)
            train_on_memory(dqn, memory)

        if done:
            break

    return reward


def train_on_episodes():
    env = Env()
    dqn = DQN(input_shape=Env.observations_shape(), output_shape=Env.actions_shape())
    memory = deque(maxlen=MEMORY_SIZE)

    reward, episode = 0, 0
    while True:
        episode += 1
        r = train_on_episode(env, dqn, memory)
        reward += (r - reward) * REWARD_DECAY
        print('Episode:', episode, 'reward:', reward)


if __name__ == '__main__':
    train_on_episodes()
