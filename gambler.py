#!/usr/bin/env python3

import numpy
import matplotlib.pyplot as plt

PROBABILITY = 0.25
MAX_SUM = 100
EPS = 1e-9

def get_possible_bets(capital):
    if capital == 0 or capital == MAX_SUM:
        return [0]
    return range(1, min(capital, MAX_SUM - capital) + 1)


class Policy:
    def __init__(self):
        self.policy = [get_possible_bets(capital) for capital in range(0, MAX_SUM + 1)]

    def __repr__(self):
        return ', '.join(map(str, self.policy))

    def get_bets(self, capital):
        return self.policy[capital]

    def improve(self, values):
        policy = self.policy
        changed = False

        for capital in range(1, MAX_SUM):
            best_value = -1
            best_bets = None

            for bet in get_possible_bets(capital):
                win = values.get_value(capital + bet)
                loss = values.get_value(capital - bet)
                value = PROBABILITY * win + (1.0 - PROBABILITY) * loss

                if value > best_value + EPS:
                    best_value = value
                    best_bets = [bet]
                elif abs(value - best_value) <= EPS:
                    best_bets.append(bet)

            if policy[capital] != best_bets:
                changed = True

            policy[capital] = best_bets

        return changed


class Values:
    def __init__(self):
        self.values = [0] * MAX_SUM
        self.values.append(1)

    def __repr__(self):
        return ', '.join(map(str, self.values))

    def get_value(self, capital):
        return self.values[capital]

    def improve(self, policy):
        while self.improve_single(policy) >= EPS:
            pass

    def improve_single(self, policy):
        values = self.values

        delta = 0

        for capital in range(1, MAX_SUM):
            value = 0
            bets = policy.get_bets(capital)
            for bet in bets:
                win = values[capital + bet]
                loss = values[capital - bet]
                value += PROBABILITY * win + (1.0 - PROBABILITY) * loss
            value /= len(bets)

            delta = max(delta, abs(values[capital] - value))
            values[capital] = value

        return delta


if __name__ == '__main__':
    policy = Policy()
    values = Values()

    iterations = 0
    while True:
        iterations += 1
        values.improve(policy)
        changed = policy.improve(values)
        if not changed:
            break

    print(policy.improve(values))
    print(values.improve(policy))

    print('Iterations passed:', iterations)
    print('Policy:', policy)
    print('Values:', values)

    plt.step(list(range(0, MAX_SUM + 1)), list(map(min, policy.policy)))
    plt.step(list(range(0, MAX_SUM + 1)), list(map(max, policy.policy)))
    plt.xlabel('Capital')
    plt.ylabel('Stake')
    plt.show()
