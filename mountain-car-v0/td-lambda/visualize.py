#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import *


def go(model_path):
    q_table = np.load(model_path)

    minP, maxP = BOUNDS[0]
    minV, maxV = BOUNDS[1]
    numP, numV = BUCKETS

    vs = []
    for v in np.linspace(minV, maxV, num=numV):
        for p in np.linspace(minP, maxP, num=numP):
            s = get_state_num([p, v])
            vs.append(np.amax(q_table[s]))
    vs = np.array(vs)
    vs = vs.reshape(numP, numV)
    plt.imshow(vs, origin='lower')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('V(position, velocity)')
    plt.xticks(np.linspace(0, numP, num=10), np.linspace(minP, maxP, num=10))
    plt.yticks(np.linspace(0, numV, num=10), np.linspace(minV, maxV, num=10))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, default='model.npy',
                        help='path to save model')
    args = parser.parse_args()
    go(args.model_path)
