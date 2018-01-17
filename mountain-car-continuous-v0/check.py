#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import tensorflow as tf

from actor_critic import ActorCritic
from utils import *


def play_episode(env, actor, render, sess):
    s = env.reset()
    if render:
        env.render()

    steps = 0
    while True:
        steps += 1
        a = actor.get_action(s, sess)[0]
        s, _, done = env.step(a)
        if render:
            env.render()

        if done:
            break
    return steps


def go(logger, render):
    env = Env()
    actor = ActorCritic(env, DISCOUNT)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, 'model.ckpt')
        logger.info('Model restored')

        steps = []
        for episode in range(STEPS_TO_WIN):
            steps.append(play_episode(env, actor, render, sess))

        logger.info('Mean score over {} episodes: {:.2f}'.format(STEPS_TO_WIN, np.mean(steps)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--render', default=False, action='store_true',
                        help='visualize')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    go(logger, args.render)
