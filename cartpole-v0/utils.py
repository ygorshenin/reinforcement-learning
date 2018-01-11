import math
import scipy.stats


BOUNDS = [(-4.9, +4.9), (-5, +5), (-0.42, +0.42), (-4, +4)]

BUCKETS = [10, 10, 10, 10]

def clamp(x, a, b):
    assert a <= b

    if x < a:
        x = a
    if x > b:
        x = b
    return x


def get_bucket(value, buckets):
    return clamp(int(value * buckets), 0, buckets - 1)


def get_state_num(observation):
    state = 0
    for i, x in enumerate(observation):
        (lo, hi) = BOUNDS[i]
        x = clamp(x, lo, hi)
        x = clamp((x - lo) / (hi - lo), 0.0, 1.0)
        buckets = BUCKETS[i]
        bucket = get_bucket(x, buckets)
        state = state * buckets + bucket
    return state


def env_reset(env):
    observation = env.reset()
    return get_state_num(observation)


def env_step(env, a):
    observation, reward, done, _ = env.step(a)
    return get_state_num(observation), reward, done
