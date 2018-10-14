#!/usr/bin/env python3

import argparse
import tensorflow as tf

from ac import AC
from env import Env

REWARD_DECAY = 0.99


def train_on_episode(sess, env, ac):
    s = env.reset()
    reward = 0

    while True:
        a, p = ac.get_action_prob(sess, s)
        s_, r, done = env.step(a)
        d = abs(r) > 1e-5
        ac.on_reward(s, a, p, r, s_, d)
        s = s_

        if d:
            reward += r
            if r > 0:
                print('Win :)')
            ac.train(sess, lr_policy=1e-3, lr_value=1e-3)

        if done:
            break

    return reward


def train_on_episodes(args):
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1

    with tf.Session(config=config) as sess:
        env = Env(render=False)
        ac = AC()

        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(args.summary, sess.graph)
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
