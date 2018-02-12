#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import tensorflow as tf

from agent import Agent
from dqn import DQN
from env import Env
from schedule import LogSchedule


DEFAULT_MAX_LR = 1e-3
DEFAULT_MIN_LR = 1e-6

DEFAULT_MAX_EPS = 1
DEFAULT_MIN_EPS = 1e-4


EPISODES_TO_WIN = 100
MEAN_REWARD_TO_WIN = 200
EPISODES_TO_SAVE = 100

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def learn_episode(agent, env):
    s = env.reset()

    reward = 0
    steps = 0

    while True:
        steps += 1

        a = agent.get_action(s)
        s_, r, done  = env.step(agent.get_action(s))

        agent.on_reward(s, a, r, s_, done)
        reward += r
        s = s_

        if done:
            break

    return steps, reward


def learn(args):
    with tf.Session() as sess:
        eps_schedule = LogSchedule(maxT=args.max_eps, minT=args.min_eps)
        lr_schedule = LogSchedule(maxT=args.max_lr, minT=args.min_lr)

        agent = Agent(sess, eps_schedule, lr_schedule)
        env = Env(render=args.render)

        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(args.summary, sess.graph)
        saver = tf.train.Saver()

        rewards = []
        solved = False
        for episode in range(args.episodes):
            progress = episode / args.episodes
            eps_schedule.on_progress(progress)
            lr_schedule.on_progress(progress)

            steps, reward = learn_episode(agent, env)
            rewards.append(reward)
            mean = np.mean(rewards[-EPISODES_TO_WIN:])

            summary = tf.Summary(value=[tf.Summary.Value(tag='running mean', simple_value=mean),
                                        tf.Summary.Value(tag='reward', simple_value=reward)])
            writer.add_summary(summary)

            logger.info('Episode {}: steps: {}, reward: {:.2f}, running mean: {:.2f}'.format(
                episode, steps, reward, mean))

            if episode % EPISODES_TO_SAVE == 0:
                path = saver.save(sess, args.model)
                logger.info('Model saved to {}'.format(path))

            if mean >= MEAN_REWARD_TO_WIN:
                logger.info('Solved! :)')
                solved = True
                break

        if not solved:
            logger.info('Not solved... :(')

        path = saver.save(sess, args.model)
        logger.info('Model saved to {}'.format(path))


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This script learns model for LunarLander-v2 environment')
    argparse.add_argument('--episodes', type=int, default=20000,
                          help='number of episodes')
    argparse.add_argument('--render', default=False, action='store_true',
                          help='render learning process')
    argparse.add_argument('--summary', type=str, default='/tmp/lunar-lander',
                          help='path to save summary')
    argparse.add_argument('--model', type=str, default='model.ckpt',
                          help='path to save model')

    argparse.add_argument('--max_lr', type=float, default=DEFAULT_MAX_LR,
                          help='maximum learning rate')
    argparse.add_argument('--min_lr', type=float, default=DEFAULT_MIN_LR,
                          help='minimum learning rate')
    argparse.add_argument('--max_eps', type=float, default=DEFAULT_MAX_EPS,
                          help='maximum probability of exploration')
    argparse.add_argument('--min_eps', type=float, default=DEFAULT_MIN_EPS,
                          help='minimum probability of exploration')

    args = argparse.parse_args()
    learn(args)
