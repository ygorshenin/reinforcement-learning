#!/usr/bin/env python3

import argparse
import tensorflow as tf

from env import Env
from pg import PG


DISCOUNT = 0.99


def train_on_episode(sess, env, episode, pg, writer):
    s, reward, steps = env.reset(), 0, 0
    memory = []

    while True:
        a = pg.get_action(sess, s)
        s_, r, done = env.step(a)
        d = abs(r) > 1e-5
        memory.append([s, a])
        s = s_
        steps += 1

        if d:
            summary = tf.Summary()
            summary.value.add(tag='reward', simple_value=r)
            summary.value.add(tag='steps', simple_value=steps)
            writer.add_summary(summary, episode)
            reward += r

            g = 1
            for [s_, a_] in reversed(memory):
                r_ = r * g
                pg.on_reward(s_, a_, r_)
                g *= DISCOUNT

            pg.train(sess, lr_policy=1e-4, beta=0.01)
            memory = []
            pg.clear_memory()
            steps = 0

        if done:
            break

    return reward


def train_on_episodes(args):
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1

    with tf.Session(config=config) as sess:
        env = Env(render=False)
        pg = PG()

        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(args.summary, sess.graph)

        saver = tf.train.Saver()

        if args.restore:
            saver.restore(sess, args.model_path)

        episode = 0

        while True:
            episode += 1
            reward = train_on_episode(sess, env, episode, pg, writer)

            print('Episode:', episode, 'reward:', reward)

            summary = tf.Summary()
            summary.value.add(tag='episode reward', simple_value=reward)
            writer.add_summary(summary, episode)

            saver.save(sess, args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path',
                        type=str,
                        default='model.ckpt',
                        help='Path to save model')
    parser.add_argument('--summary',
                        type=str,
                        default='/tmp/pong-v0',
                        help='Path to save summary')
    parser.add_argument('--restore',
                        action='store_true',
                        help='When true model will be pre-loaded')
    args = parser.parse_args()
    train_on_episodes(args)
