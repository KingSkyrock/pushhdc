"""Tests for the random genes that `encoder_spawner` makes."""

import numpy as np

from pushhdc.encoder_ops import encoder_instructions, encoder_spawner
from pushhdc.evaluation import Evaluator

# Tests


def test_samples_of_one_length_get_the_ints_of_a_plain_draw() -> None:
    """when all samples have the same length, a seed gives the same ints as
    plain draws from 0 to the last unit.

    fails if the draw still chooses a half there. choosing a half uses more
    random numbers, which would change every seeded run on samples of one
    length.
    """
    spawner = encoder_spawner(
        encoder_instructions(unit_ops=True),
        1,
        unit_ops=True,
        seen_lengths=np.full(4, 8),
    )
    # the int draw is the only maker of random ints that the spawner has
    draw = spawner.erc_generators[0]
    rng, plain = np.random.default_rng(0), np.random.default_rng(0)
    assert [draw(rng) for _ in range(50)] == [
        int(plain.integers(0, 8)) for _ in range(50)
    ]


def test_ints_fit_short_samples_and_still_reach_the_longest() -> None:
    """when most samples are much shorter than the longest, many new ints
    fall inside a short sample, and many still reach the far end of the
    longest sample.

    fails if every int is drawn from 0 to the last unit, because then few
    ints fit a short sample. fails if every int is a place in a sample,
    because then few ints reach the far end. also fails if a place can be
    one unit past the end of its sample, if the draw reads the lengths of
    test samples, or if empty samples are not left out, because an empty
    sample has no place to draw.
    """
    # evolution sees empty samples, samples of one unit and one long sample.
    # the only place in a sample of one unit is 0. the test samples are all
    # long, so a place in them is rarely 0
    lengths = np.array([0] * 10 + [1] * 9 + [100] * 81)
    evaluator = Evaluator(
        [np.ones((len(lengths), 100, 1))],
        np.zeros(len(lengths)),
        np.arange(20),
        seed=0,
        unit_ops=True,
        lengths=lengths,
    )
    # the int draw is the only maker of random ints that the spawner has
    draw = evaluator.spawner().erc_generators[0]
    rng = np.random.default_rng(0)
    ints = np.array([draw(rng) for _ in range(2000)])
    # about half of the ints are 0, and about a quarter are in the far half
    # of the long sample. each bug that the docstring names makes one of these
    # two shares much smaller
    assert np.mean(ints == 0) > 0.3
    assert np.mean(ints >= 50) > 0.15
