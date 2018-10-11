import gym
import numpy as np


MAX_FRAMES = 2

MIN_ROW = 33
MAX_ROW = 194
NROWS = MAX_ROW - MIN_ROW

MIN_COL = 0
MAX_COL = 160
NCOLS = MAX_COL - MIN_COL


class Env:
    def __init__(self, render=False):
        self.render = render
        self.env = gym.make('Pong-v0')

    def reset(self):
        frame = self.env.reset()
        frame = Env._normalize_frame(frame)
        if self.render:
            self.env.render()

        self.f0, self.f1 = frame, frame
        self.state = np.vstack([self.f0, self.f1])
        return self.state

    def step(self, action):
        frame, reward, done, _ = self.env.step(2 + action)
        frame = Env._normalize_frame(frame)

        self.f0, self.f1 = self.f1, frame
        self.state = np.vstack([self.f0, self.f1])

        if self.render:
            self.env.render()
        return self.state, reward, done

    @staticmethod
    def actions_dim():
        return 2

    @staticmethod
    def observations_shape():
        return (MAX_FRAMES * NROWS, NCOLS, 1)

    @staticmethod
    def _normalize_frame(frame):
        frame = frame[MIN_ROW:MAX_ROW, :, 0]
        frame[frame == 144] = 0
        frame[frame != 0] = 1
        return frame.reshape([NROWS, NCOLS, 1])
