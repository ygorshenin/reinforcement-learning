import gym
import numpy as np
import time


class Env:
    def __init__(self, render=False):
        self.render = render
        self.env = gym.make('Pong-v0')

    def reset(self):
        frame = self.env.reset()
        frame = Env._normalize_frame(frame)
        if self.render:
            self.env.render()
            time.sleep(0.02)

        self.f0, self.f1 = frame, frame
        return self.make_state()

    def step(self, action):
        frame, reward, done, _ = self.env.step(2 + action)
        if self.render:
            self.env.render()
            time.sleep(0.02)

        frame = Env._normalize_frame(frame)
        self.f0, self.f1 = self.f1, frame

        return self.make_state(), reward, done

    @staticmethod
    def observations_shape():
        return [1, 80, 160]

    @staticmethod
    def _normalize_frame(frame):
        frame = frame[34:193, :, 0]
        frame[frame == 53] = 0
        frame[frame == 109] = 0
        frame[frame == 144] = 0
        frame[frame != 0] = 1
        return frame[::2, ::2]

    def make_state(self):
        return np.expand_dims(np.hstack([self.f0, self.f1]), axis=0)
