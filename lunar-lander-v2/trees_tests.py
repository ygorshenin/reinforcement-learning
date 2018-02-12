#!/usr/bin/env python3
import pytest

from trees import *


def test_max_smoke():
    tree = MaxTree(0)
    assert tree.max() == 0


def test_max_usage():
    tree = MaxTree(4)

    assert tree.max() == 0

    tree.set(1, 10)
    assert tree.max() == 10
    assert all(tree.leaves() == [0, 10, 0, 0])

    tree.set(0, 5)
    assert tree.max() == 10
    assert all(tree.leaves() == [5, 10, 0, 0])

    tree.set(1, 4)
    assert tree.max() == 5
    assert all(tree.leaves() == [5, 4, 0, 0])

    tree.set(2, 3)
    assert tree.max() == 5
    assert all(tree.leaves() == [5, 4, 3, 0])

    tree.set(0, 1);
    tree.set(1, 2)
    assert tree.max() == 3
    assert all(tree.leaves() == [1, 2, 3, 0])


def test_prob_smoke():
    tree = ProbTree(0)


def test_prob_usage():
    tree = ProbTree(3)

    tree.set(0, 10)
    assert tree.probs() == [1, 0, 0]

    assert tree.sample_1() == 0
    assert tree.sample_n(1) == [0]

    assert tree.probs() == [1, 0, 0]

    print(tree.buffer)
    tree.set(2, 10)
    assert tree.probs() == [0.5, 0, 0.5]

    sample = tree.sample_1()
    assert sample == 0 or sample == 2

    samples = tree.sample_n(2)
    assert sorted(samples) == [0, 2]

    tree.set(1, 5)
    assert tree.probs() == [0.4, 0.2, 0.4]

    tree.set(0, 0)
    assert tree.probs() == [0, 1 / 3, 2 / 3]
