import gym
import numpy as np


DISCOUNT = 1.0
STEPS_TO_WIN = 100


class Env:
    def __init__(self):
        self.env = gym.make('MountainCarContinuous-v0')

        observation_space = self.env.observation_space
        self.states_dim = observation_space.shape[0]
        self.low = observation_space.low
        self.high = observation_space.high

        action_space = self.env.action_space
        self.min_action = action_space.low[0]
        self.max_action = action_space.high[0]

    def reset(self):
        return self._normalize_state(self.env.reset())

    def render(self):
        self.env.render()

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return self._normalize_state(state), reward, done

    def _normalize_state(self, state):
        state = 2 * (state - self.low) / (self.high - self.low) - 1
        return np.reshape(state, [1, self.states_dim])
