#!/usr/bin/env python3

from keras.models import load_model
import argparse

from utils import *


def go(resolution, model_path):
    model = load_model(model_path)
    show_dqn(resolution, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resolution', type=int, default=1000,
                        help='single dimension of an image')
    parser.add_argument('--model_path', type=str, default='model.h5y',
                        help='path to load model')
    args = parser.parse_args()
    go(args.resolution, args.model_path)
