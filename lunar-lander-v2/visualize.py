#!/usr/bin/env python3

import argparse
import logging
import tensorflow as tf

from agent import Agent
from env import *
from schedule import ConstSchedule


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def visualize_episode(agent, env):
    s = env.reset()
    reward = 0
    while True:
        a = agent.get_action(s)
        s, r, done = env.step(a)
        reward += r
        if done:
            break
    return reward


def visualize(args):
    with tf.Session() as sess:
        eps_schedule = ConstSchedule(T=0)
        lr_schedule = None

        agent = Agent(sess, eps_schedule, lr_schedule)
        env = Env(render=True)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, args.model)

        while True:
            reward = visualize_episode(agent, env)
            logger.info('Reward: {:.2f}'.format(reward))


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This script visualizes trained model behavior')
    argparse.add_argument('--model', type=str, default='model.ckpt',
                          help='path to load model')
    args = argparse.parse_args()
    visualize(args)
