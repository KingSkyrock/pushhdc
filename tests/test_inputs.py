"""Tests for the inputs of an evaluator: shared inputs, samples of many
lengths and `Lookup` inputs."""

import pickle

import numpy as np
import pytest

from pushhdc.inputs import Lookup
from tests.helpers import (
    PROGRAMS_FOR_LENGTHS,
    evaluator_on,
    genome,
    samples_of_many_lengths,
)

# Tests


def test_a_shared_input_scores_like_a_copied_one() -> None:
    """an input with 1 sample gives the same hvs and fingerprints as the same
    input copied to every sample, in a program that uses it with another
    input and in a program that uses it alone.

    fails if the probes take samples from the shared input, or if a program
    that only uses the shared input gives 1 hv in place of 1 hv per sample.
    """
    rng = np.random.default_rng(0)
    a = rng.choice([-1, 1], size=(3, 4, 8))
    b = rng.choice([-1, 1], size=(1, 4, 8))
    shared = evaluator_on(a, b, unit_ops=True)
    copied = evaluator_on(a, np.repeat(b, 3, axis=0), unit_ops=True)
    for program in (
        genome("input_0", "input_1", "hv_bind"),
        genome("input_1", 1, "hv_shift_units"),
    ):
        hvs, _ = shared.encode(program)
        copied_hvs, _ = copied.encode(program)
        assert hvs is not None
        assert copied_hvs is not None
        assert np.array_equal(hvs, copied_hvs)
        assert shared.fingerprint(program) == copied.fingerprint(program)


def test_each_sample_encodes_as_if_alone() -> None:
    """with `lengths`, each sample gets the hv and the program gets the cost
    that they get with that sample alone, cut to its own length.

    fails if units past a sample's length count, if a shared input is not cut
    to each length, if hvs come back in the wrong order, or if the program
    pays its cost one time for each length.
    """
    a, b, lengths = samples_of_many_lengths()
    together = evaluator_on(a, b, unit_ops=True, lengths=lengths)
    for program in PROGRAMS_FOR_LENGTHS:
        hvs, cost = together.encode(program)
        assert hvs is not None
        for i, length in enumerate(lengths):
            alone = evaluator_on(
                a[i : i + 1, :length], b[:, :length], unit_ops=True
            )
            alone_hvs, alone_cost = alone.encode(program)
            assert alone_hvs is not None
            assert np.array_equal(hvs[i], alone_hvs[0])
            assert cost == alone_cost


def test_a_lookup_input_scores_like_the_array_it_stands_for() -> None:
    """a `Lookup` gives the same hvs, cost and fingerprint as the array of
    each unit's hv, for samples of many lengths and for a shared `Lookup`.

    fails if a cut takes the wrong samples or units, if `shape` gives the
    wrong number of units, or if a `Lookup` with one row is not cut as a
    shared input.
    """
    rng = np.random.default_rng(0)
    letters = rng.choice([-1, 1], size=(5, 16))
    symbols = rng.integers(0, 5, size=(6, 8))
    places = rng.choice([-1, 1], size=(8, 16))
    _, _, lengths = samples_of_many_lengths()
    looked_up = evaluator_on(
        Lookup(letters, symbols),
        Lookup(places, np.arange(8)[None]),
        unit_ops=True,
        lengths=lengths,
    )
    arrays = evaluator_on(
        letters[symbols], places[None], unit_ops=True, lengths=lengths
    )
    assert Lookup(letters, symbols).shape == letters[symbols].shape
    for program in PROGRAMS_FOR_LENGTHS:
        hvs, cost = looked_up.encode(program)
        array_hvs, array_cost = arrays.encode(program)
        assert hvs is not None
        assert array_hvs is not None
        assert np.array_equal(hvs, array_hvs)
        assert cost == array_cost
        assert looked_up.fingerprint(program) == arrays.fingerprint(program)


def test_lookup_rejects_symbols_below_zero() -> None:
    """a symbol below 0 raises an error.

    fails if -1 picks the last hv with no error, as numpy indexing does.
    """
    with pytest.raises(ValueError, match="symbols must pick one of"):
        Lookup(np.ones((3, 4)), np.array([[0, -1]]))


def test_a_lookup_input_keeps_the_evaluator_small() -> None:
    """an evaluator holds the symbols of a `Lookup` input, not the hv of each
    unit, so the copy that each worker process gets stays small.

    fails if the evaluator makes the hvs of a `Lookup` when it cuts it.
    """
    rng = np.random.default_rng(0)
    hvs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, 1000))
    symbols = rng.integers(0, 4, size=(200, 50)).astype(np.int8)
    evaluator = evaluator_on(Lookup(hvs, symbols))
    # the hv of every unit takes ten times this many bytes
    assert len(pickle.dumps(evaluator)) < 1_000_000
