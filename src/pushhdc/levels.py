"""Hvs for numbers, so that an input can hold numbers as well as symbols.

Each number gets one of a few levels, and each level has an hv. Near levels
have alike hvs, so near numbers get alike hvs. The levels of an input are
symbols for a `Lookup` (see `pushhdc.inputs`), with the level hvs as its hvs.
"""

import numpy as np


def level_hvs(
    count: int, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    """one hv of -1 and 1 for each of `count` levels, shaped (count,
    dimension). the first and the last hv are random, so they are not alike.
    each dimension changes from its value in the first hv to its value in the
    last hv at a random level. so two levels are less alike the farther apart
    they are, and the similarity falls in a straight line."""
    if count < 2:
        raise ValueError("there must be 2 or more levels")
    first, last = rng.choice(np.array([-1, 1], dtype=np.int8), (2, dimension))
    # each dimension has the value of the last hv from this level on
    changes = rng.integers(1, count, size=dimension)
    hvs: np.ndarray = np.where(
        changes <= np.arange(count)[:, None], last, first
    )
    return hvs


def levels(values: np.ndarray, seen: np.ndarray, count: int) -> np.ndarray:
    """the level of each number in `values`, from 0 to `count` - 1. the levels
    cut `seen` into parts of about the same size, so the levels are narrow
    where numbers are common. when many numbers in `seen` are the same, they
    all get one level, and some levels can stay empty. a number below or
    above all of `seen` gets the first or the last level.

    `seen` should be the numbers of the samples that evolution sees, so that
    the test samples do not change the levels."""
    if count < 2:
        raise ValueError("there must be 2 or more levels")
    edges = np.quantile(seen, np.arange(1, count) / count)
    return np.searchsorted(edges, values, side="right").astype(
        np.min_scalar_type(count - 1)
    )
