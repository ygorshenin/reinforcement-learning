import gym
import matplotlib.pyplot as plt
import numpy as np


class Env:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        observation_space = self.env.observation_space

        self.states_dim = observation_space.shape[0]

        # We omit stop action (1), and leave only move left (0) and
        # move right (2).  This gives us much better learning without
        # any loss of quality.
        self.actions_dim = 2
        self.bounds = list(zip(observation_space.low, observation_space.high))

    def reset(self):
        state = self.env.reset()
        return self._to_vector(state)

    def step(self, action):
        state, reward, done, _ = self.env.step(2 * action)
        return self._to_vector(state), reward, done

    def render(self):
        return self.env.render()

    def _to_vector(self, state):
        vs = np.zeros(self.states_dim)
        for i, (low, high) in enumerate(self.bounds):
            vs[i] = 2 * (state[i] - low) / (high - low) - 1
            assert vs[i] >= -1
            assert vs[i] <= +1
        return vs.reshape((1, self.states_dim))


def dqn_image(resolution, model):
    position = np.linspace(-1, 1, resolution)
    velocity = np.linspace(-1, 1, resolution)

    xs = []
    for v in np.linspace(-1, 1, resolution):
        for p in np.linspace(-1, 1, resolution):
            xs.append([p, v])
    xs = np.array(xs)
    ys = model.predict(xs)
    vs = np.amax(ys, axis=1)
    return vs.reshape(resolution, resolution)


def show_dqn(resolution, model):
    vs = dqn_image(resolution, model)
    plt.title('V(position, velocity)')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.imshow(vs, origin='lower')
    plt.show()


def save_dqn(resolution, model, path):
    vs = dqn_image(resolution, model)
    plt.imsave(fname=path, arr=vs, origin='lower')
