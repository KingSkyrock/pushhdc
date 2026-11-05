import numpy as np

from pushhdc.hypervector import HyperVector


def test_random_hypervector_is_reproducible_with_seed() -> None:
    """the same seed gives the same random hypervector.

    fails if `random` draws from its own unseeded generator instead of
    `rng`.
    """
    first = HyperVector.random(1000, rng=np.random.default_rng(0))
    again = HyperVector.random(1000, rng=np.random.default_rng(0))
    assert np.array_equal(first.data, again.data)
