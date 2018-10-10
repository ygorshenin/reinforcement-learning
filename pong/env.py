import gym
import numpy as np

from collections import deque


MAX_FRAMES = 4

_MIN_ROW = 33
_MAX_ROW = 194

_MIN_COL = 0
_MAX_COL = 160


class Env:
    def __init__(self, render=False):
        self.render = render
        self.env = gym.make('Pong-v0')

    def reset(self):
        frame = self.env.reset()
        frame = Env._normalize_frame(frame)
        if self.render:
            self.env.render()
        self.state = np.vstack([frame, frame, frame, frame])
        return self.state

    def step(self, action):
        frame, reward, done, _ = self.env.step(2 + action)
        frame = Env._normalize_frame(frame)
        self.state = np.vstack([frame, self.state])
        self.state = self.state[:MAX_FRAMES * (_MAX_ROW - _MIN_ROW), :, :]

        if self.render:
            self.env.render()
        return self.state, reward, done

    @staticmethod
    def actions_shape():
        return 2

    @staticmethod
    def observations_shape():
        return (MAX_FRAMES * (_MAX_ROW - _MIN_ROW), _MAX_COL - _MIN_COL, 1)

    @staticmethod
    def _normalize_frame(frame):
        frame = frame[_MIN_ROW:_MAX_ROW, :, 0:1] / 255
        return frame
