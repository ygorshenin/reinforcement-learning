import gym
import numpy as np


STATES_DIM = 8
ACTIONS_DIM = 4


class Env:
    """Env wrapper for LunarLander-v2 discrete version.

       State: unknown
       Actions: [nope, main engine, left engine, right engine]
    """

    def __init__(self, render):
        self._env = gym.make('LunarLander-v2')
        self.render = render

    def reset(self):
        state = self._env.reset()
        if self.render:
            self._env.render()
        return self._normalize_state(state)

    def step(self, action):
        state, reward, done, _ = self._env.step(action)
        if self.render:
            self._env.render()
        return self._normalize_state(state), reward, done

    @staticmethod
    def _normalize_state(state):
        return np.array(state)
