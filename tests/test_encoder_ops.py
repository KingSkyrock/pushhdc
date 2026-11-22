"""Tests for the random genes that `encoder_spawner` makes."""

import numpy as np

from pushhdc.encoder_ops import encoder_instructions, encoder_spawner

# Tests


def test_samples_of_one_length_get_the_ints_of_a_plain_draw() -> None:
    """when all samples have the same length, a seed gives the same ints as
    plain draws from 0 to the last unit.

    fails if the draw still chooses a half there. choosing a half uses more
    random numbers, so a seed no longer gives the same results as before on
    samples of one length.
    """
    spawner = encoder_spawner(
        encoder_instructions(unit_ops=True),
        1,
        unit_ops=True,
        seen_lengths=np.full(4, 8),
    )
    draw = spawner.erc_generators[0]
    rng, plain = np.random.default_rng(0), np.random.default_rng(0)
    assert [draw(rng) for _ in range(50)] == [
        int(plain.integers(0, 8)) for _ in range(50)
    ]
