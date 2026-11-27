"""Tests for the levels and level hvs that let an input hold numbers."""

import numpy as np
import pytest

from pushhdc.levels import level_hvs, levels

# Helpers


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(a.astype(float) @ b / len(a))


# Tests


def test_level_hvs_get_less_alike_in_a_straight_line() -> None:
    """the similarity of two levels is 1 minus their distance over the
    distance from the first to the last level, so the first and the last are
    not alike. each value is -1 or 1.

    fails if the similarity does not fall in a straight line, for example
    when every dimension changes at the same level, or if the first or the
    last hv is not the end of the line.
    """
    hvs = level_hvs(11, 40_000, np.random.default_rng(0))
    assert set(np.unique(hvs)) == {-1, 1}
    for i in range(11):
        for j in range(11):
            assert similarity(hvs[i], hvs[j]) == pytest.approx(
                1 - abs(i - j) / 10, abs=0.02
            )


def test_levels_cut_the_seen_numbers_into_equal_parts() -> None:
    """each level holds the same share of `seen`, also when the numbers are
    far from evenly spread, and a bigger number never gets a lower level.
    numbers outside `seen` get the first or the last level.

    fails if the levels cut the range of the numbers evenly, if there are not
    exactly `count` levels, if a number outside `seen` gets a level outside
    0 to `count` - 1, or if the levels come from the numbers to level and not
    from `seen`.
    """
    seen = np.random.default_rng(0).exponential(size=10_000)
    got = levels(seen, seen, 10)
    assert np.array_equal(np.bincount(got), np.full(10, 1000))
    assert np.all(np.diff(got[np.argsort(seen)]) >= 0)
    assert levels(np.array([-1e9, 1e9]), seen, 10).tolist() == [0, 9]
    assert np.array_equal(levels(seen[::7], seen, 10), got[::7])
