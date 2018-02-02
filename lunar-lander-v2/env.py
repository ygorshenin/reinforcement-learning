import gym
import numpy as np


STATES_DIM = 8
ACTIONS_DIM = 4

MEANS = np.array([ 0.0137619, 0.60699932, 0.04854489, -0.85873458, -0.19234456, -0.11691365, 0.03115791, 0.0194389 ])
STDS = np.array([ 0.16609065, 0.31256512, 0.39388902, 0.54769906, 0.67717174, 0.82077772, 0.17374435, 0.13806168])

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
        return (np.array(state) - MEANS) / STDS
