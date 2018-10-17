#!/usr/bin/env python3

import argparse
import tensorflow as tf

from ac import AC
from env import Env


REWARD_DECAY = 0.1
DISCOUNT = 0.99


def train_on_episode(sess, env, ac):
    s, reward = env.reset(), 0

    memory = []

    while True:
        a = ac.get_action(sess, s)
        s_, r, done = env.step(a)
        d = abs(r) > 1e-5
        memory.append([s, a, r, s_, d])
        s = s_

        if d:
            reward += r
            if r > 0:
                print('Win :)')
            else:
                print('Lose :(')

            norm = max(len(memory), 1)
            for row in reversed(memory):
                row[2] = r / norm
                ac.on_reward(*row)
                r *= DISCOUNT

            memory = []
            ac.train(sess, lr_policy=1e-4, lr_value=1e-4)
            ac.clear_memory()

        if done:
            break

    return reward


def train_on_episodes(args):
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1

    with tf.Session(config=config) as sess:
        env = Env(render=False)
        ac = AC(discount=DISCOUNT)

        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(args.summary, sess.graph)
        ac.writer = writer

        saver = tf.train.Saver()

        if args.restore:
            saver.restore(sess, args.model_path)

        reward, episode = 0, 0

        while True:
            episode += 1
            r = train_on_episode(sess, env, ac)
            reward += (r - reward) * REWARD_DECAY

            summary = tf.Summary(value=[tf.Summary.Value(tag='running mean', simple_value=reward),
                                        tf.Summary.Value(tag='reward', simple_value=reward)])
            writer.add_summary(summary)

            print('Episode:', episode, 'reward:', reward)
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
