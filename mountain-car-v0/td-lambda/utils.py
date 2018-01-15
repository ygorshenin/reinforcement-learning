import gym


BOUNDS = [(-1.2, +0.6), (-0.07, +0.07)]
BUCKETS = [64, 64]


def clamp(x, low, high):
    assert low <= high
    if x < low:
        x = low
    if x > high:
        x = high
    return x


def get_bucket(x, buckets):
    return clamp(int(x * buckets), 0, buckets - 1)


def get_state_num(observation):
    state = 0
    for i, x in enumerate(observation):
        (lo, hi) = BOUNDS[i]
        assert x >= lo
        assert x <= hi

        buckets = BUCKETS[i]
        x = clamp((x - lo) / (hi - lo), 0, 1)

        bucket = get_bucket(x, buckets)
        state = state * buckets + bucket
    return state


class Env:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')

    def reset(self):
        observation = self.env.reset()
        return get_state_num(observation)

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        return get_state_num(observation), reward, done

    def render(self):
        return self.env.render()
