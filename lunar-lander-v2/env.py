import gym
import numpy as np


STATES_DIM = 8
ACTIONS_DIM = 4


class Env:
    """Env wrapper for LunarLander-v2 discrete.

       State: unknown
       Actions: [nope, main engine, left engine, right engine]
    """

    def __init__(self, render):
        self._env = gym.make('LunarLander-v2')
        if render:
            self._env = gym.wrappers.Monitor(self._env, 'videos', video_callable=lambda episode_id: episode_id % 100 == 0)

    def reset(self):
        state = self._env.reset()
        return self._normalize_state(state)

    def step(self, action):
        state, reward, done, _ = self._env.step(action)
        return self._normalize_state(state), reward, done

    @staticmethod
    def _normalize_state(state):
        return np.array(state)
