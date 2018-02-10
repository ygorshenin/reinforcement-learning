#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import tensorflow as tf

from agent import Agent
from dqn import DQN
from env import Env


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
        agent = Agent(sess)
        env = Env(render=args.render)

        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(args.summary, sess.graph)
        saver = tf.train.Saver()

        rewards = []
        solved = False
        for episode in range(args.episodes):
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
    args = argparse.parse_args()
    learn(args)
