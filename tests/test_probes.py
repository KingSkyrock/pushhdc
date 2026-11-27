"""Tests for the fingerprints and kernels that an evaluator gets from running
a program on a few probe samples."""

import numpy as np
import pytest

from tests.helpers import (
    PROGRAMS_FOR_LENGTHS,
    evaluator_on,
    genome,
    samples_of_many_lengths,
)

# Tests


def test_fingerprint_matches_programs_that_encode_alike() -> None:
    """programs share a fingerprint when they encode samples the same way, and
    each still gets its own cost.

    fails if the fingerprint depends on anything but the encoding, for
    example the program's cost, or if it misses a difference between
    encodings.
    """
    a, b = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 3, 4, 16))
    evaluator = evaluator_on(a, b)
    plain, plain_cost = evaluator.fingerprint(genome("input_0"))
    # binds `b` with itself, then leaves `a` on top, so it encodes like `plain`
    detour, detour_cost = evaluator.fingerprint(
        genome("input_1", "input_1", "hv_bind", "input_0")
    )
    other, _ = evaluator.fingerprint(genome("input_1"))
    assert plain == detour != other
    assert (plain_cost, detour_cost) == (0, 1)


def test_each_genome_gets_its_own_kernel_after_fingerprints() -> None:
    """after `fingerprint` runs on some genomes, `kernel` gives each of them
    the same kernel as an evaluator that made no fingerprints.

    fails if the kernels that `fingerprint` keeps for `kernel` get mixed up
    between genomes.
    """
    a, b = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 8, 3, 16))
    programs = [
        genome("input_0"),
        genome("input_1"),
        genome("input_0", "input_1", "hv_bind"),
    ]
    evaluator = evaluator_on(a, b)
    for program in programs:
        evaluator.fingerprint(program)
    for program in programs:
        kept, fresh = (
            evaluator.kernel(program),
            evaluator_on(a, b).kernel(program),
        )
        assert kept is not None
        assert fresh is not None
        assert kept.alignment(fresh) == pytest.approx(1)


def test_padding_does_not_change_fingerprints() -> None:
    """the values past each sample's length do not change a fingerprint, and
    programs that encode the samples in different ways get different
    fingerprints.

    fails if the probes for `fingerprint` keep those values, so that programs
    that encode the samples alike get different fingerprints, or if the
    probes lose the units that tell programs apart.
    """
    a, b, lengths = samples_of_many_lengths()
    other = a.copy()
    past_the_end = np.arange(a.shape[1]) >= lengths[:, None]
    other[past_the_end] = -other[past_the_end]
    first = evaluator_on(a, b, unit_ops=True, lengths=lengths)
    second = evaluator_on(other, b, unit_ops=True, lengths=lengths)
    for program in PROGRAMS_FOR_LENGTHS:
        assert first.fingerprint(program) == second.fingerprint(program)
    keys = [first.fingerprint(program)[0] for program in PROGRAMS_FOR_LENGTHS]
    assert keys[0] != keys[1]


def test_probes_see_the_longest_sample() -> None:
    """the probes for `fingerprint` include the longest sample, also when it
    is not one of the first probes.

    fails if the probes stop at a shorter length. then programs that differ
    only on the units of the longest samples get one fingerprint, and one of
    them gets the saved score of the other.
    """
    a = np.random.default_rng(0).choice([-1, 1], size=(200, 8, 16))
    short = genome("input_0", 0, 3, "hv_keep_units")
    long = genome("input_0", 0, 7, "hv_keep_units")
    # the long sample goes to many places, so some of them are not among the
    # first probes
    for place in range(0, 200, 25):
        lengths = np.full(200, 2)
        lengths[place] = 8
        evaluator = evaluator_on(a, unit_ops=True, lengths=lengths)
        assert evaluator.fingerprint(short)[0] != evaluator.fingerprint(long)[0]
