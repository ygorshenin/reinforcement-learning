#!/usr/bin/env python3

import argparse
import logging
import math
import numpy as np
import os
import tensorflow as tf

from actor_critic import ActorCritic
from utils import *


MAX_LEARNING_RATE_VALUE = 1e-4
MIN_LEARNING_RATE_VALUE = 1e-6
MAX_LEARNING_RATE_POLICY = 1e-4
MIN_LEARNING_RATE_POLICY = 1e-6


class LearningRate:
    def __init__(self, start, end, episodes):
        self.start = start
        self.end = end
        self.ratio = self.end / self.start
        self.episodes = episodes

    def get_lr(self, episode):
        progress = episode / self.episodes
        return self.start * math.pow(self.ratio, progress)


def learn_episode(env, actor, lr_policy, lr_value, render, sess):
    s = env.reset()
    steps = 0

    while True:
        if render:
            env.render()
        steps += 1

        a = actor.get_action(s, sess)[0]
        s_, r, done = env.step(a)
        actor.on_reward(s, a, r, s_, done)
        actor.train(lr_policy, lr_value, sess)
        if done:
            break
        s = s_

    return steps


def learn(logger, episodes, render):
    env = Env()
    actor = ActorCritic(env, DISCOUNT)

    lr_policy = LearningRate(MAX_LEARNING_RATE_POLICY, MIN_LEARNING_RATE_POLICY, episodes)
    lr_value = LearningRate(MAX_LEARNING_RATE_VALUE, MIN_LEARNING_RATE_VALUE, episodes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        scores = []
        for episode in range(episodes):
            lrp = lr_policy.get_lr(episode)
            lrv = lr_value.get_lr(episode)

            score = learn_episode(env, actor, lrp, lrv, render, sess)
            scores.append(score)

            mean = np.mean(scores[-STEPS_TO_WIN:])

            args = [episode, episodes, score, mean, lrp, lrv]
            logger.info('After {}/{} episodes: {}, mean: {:.2f}, lrp: {:.6f}, lrv: {:.6f}'.format(*args))

        save_path = saver.save(sess, os.path.join(os.getcwd(), 'model.ckpt'))
        logger.info('Model saved to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--episodes', type=int, default=10000,
                        help='number of episodes to play')
    parser.add_argument('--render', default=False, action='store_true',
                        help='visualize learning')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    learn(logger, args.episodes, args.render)
