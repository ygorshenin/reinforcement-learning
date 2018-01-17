#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from actor_critic import ActorCritic
from utils import *


def go(resolution):
    env = Env()
    actor = ActorCritic(env, DISCOUNT)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, 'model.ckpt')

        ps = np.linspace(-1, 1, num=resolution)
        vs = np.linspace(-1, 1, num=resolution)

        states = []
        for v in vs:
            for p in ps:
                states.append([p, v])

        states = np.reshape(np.array(states), [resolution * resolution, 2])
        values = actor.get_value(states, sess)
        values = np.reshape(values, [resolution, resolution])
        plt.imshow(values, origin='lower')

        plt.title('Value(position, velocity)')

        minx, maxx = env.low[0], env.high[0]
        xticks = map(lambda x: '{:.2f}'.format(x), np.linspace(minx, maxx, num=10))
        plt.xticks(np.linspace(0, resolution, num=10), xticks)
        plt.xlabel('Position')

        miny, maxy = env.low[1], env.high[1]
        plt.ylabel('Velocity')
        yticks = map(lambda y: '{:.2f}'.format(y), np.linspace(miny, maxy, num=10))
        plt.yticks(np.linspace(0, resolution, num=10), yticks)

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resolution', type=int, default=1000, help='resolution')

    args = parser.parse_args()
    go(args.resolution)
