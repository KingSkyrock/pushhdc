"""Tests for scoring Push programs as HDC encoders, on tiny made-up data."""

from typing import Any

import numpy as np
import pytest

from pushhdc.evaluation import Evaluator, Lookup
from pushhdc.evolution import Fitness
from pushhdc.pyshgp.gp.genome import Genome
from pushhdc.pyshgp.push.atoms import Input, InstructionMeta, Literal
from pushhdc.pyshgp.push.types import PushInt

# Helpers


# lets tests write genomes as names like "input_0" and "hv_bind". ints become
# int literals
def genome(*genes: str | int) -> Any:
    def gene(name: str | int) -> Any:
        if isinstance(name, int):
            return Literal(value=name, push_type=PushInt)
        if name.startswith("input_"):
            return Input(input_index=int(name.removeprefix("input_")))
        return InstructionMeta(name=name, code_blocks=0)

    return Genome([gene(name) for name in genes])


# for tests of one program, where the labels and folds do not matter
def encoder(
    *inputs: np.ndarray | Lookup,
    unit_ops: bool = False,
    lengths: np.ndarray | None = None,
) -> Evaluator:
    samples = np.arange(len(inputs[0]))
    labels = np.zeros(len(samples))
    return Evaluator(
        inputs,
        labels,
        samples,
        folds=2,
        seed=0,
        unit_ops=unit_ops,
        lengths=lengths,
    )


# samples with random values in every unit, so the units past each length
# hold values that must not count. the lengths are out of order, and one
# length comes twice with another length between, so samples of one length
# are not next to each other. the lengths include 0 and all the units. one
# input is shared by all samples
def samples_of_many_lengths() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    a = rng.choice([-1, 1], size=(6, 8, 16))
    b = rng.choice([-1, 1], size=(1, 8, 16))
    return a, b, np.array([3, 0, 7, 3, 8, 1])


# a shift brings the unit after the end into the last unit. a program that
# uses only the shared input gives one hv for each length, and every sample of
# that length must get it
PROGRAMS_FOR_LENGTHS = (
    genome("input_0", 1, "hv_shift_units", "input_1", "hv_bind"),
    genome("input_1"),
)


# Tests


def test_encoding_permutes_each_hv_and_bundles_units() -> None:
    """a program runs over every unit, then each sample's units get bundled.

    fails if inputs get mixed up, if permute shifts values between samples or
    units instead of within each hv, if the wrong axis gets bundled, if bind
    costs nothing, or if permute has a cost.
    """
    a, b = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 3, 4, 16))
    hvs, cost = encoder(a, b).encode(
        genome("input_1", 1, "hv_permute", "input_0", "hv_bind")
    )
    assert hvs is not None
    assert np.array_equal(hvs, (a * np.roll(b, 1, axis=-1)).sum(axis=1))
    assert cost == 1


def test_each_sample_is_scored_without_its_own_fold() -> None:
    """accuracy covers every sample, each guessed by prototypes built from the
    other folds.

    fails if a fold's own samples end up in its prototypes, if some samples
    never get scored, or if guesses are class positions instead of the labels
    themselves.
    """
    classes = np.random.default_rng(0).choice([-1.0, 1.0], size=(3, 64))
    # the only class 9 sample can't be guessed right, since its class is
    # missing from the other folds. with folds of 3, the other folds always
    # still have some 5s and 7s, so everything else gets guessed right
    labels = np.array([5, 5, 5, 5, 7, 7, 7, 7, 9])
    samples = classes[[0, 0, 0, 0, 1, 1, 1, 1, 2]]
    evaluator = Evaluator(
        [samples[:, None, :]], labels, np.arange(9), folds=3, seed=0
    )
    assert evaluator(genome("input_0")).accuracy == pytest.approx(8 / 9)


def test_program_with_no_output_scores_zero() -> None:
    """a program that leaves no hv scores 0 and still pays for its ops.

    fails if the evaluator crashes on the missing output.
    """
    a = np.ones((2, 1, 8))
    assert encoder(a)(genome("hv_bind")) == Fitness(accuracy=0.0, cost=1)


def test_bundle_adds_hvs() -> None:
    """`hv_bundle` adds two hvs and keeps the sums.

    fails if bundle errors and the interpreter skips it, which leaves the last
    input as the output, or if it rounds sums back to -1s and 1s.
    """
    a, b = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 2, 3, 16))
    hvs, _ = encoder(a, b).encode(genome("input_0", "input_1", "hv_bundle"))
    assert hvs is not None
    assert np.array_equal(hvs, (a + b).sum(axis=1))


def test_fingerprint_matches_programs_that_encode_alike() -> None:
    """programs share a fingerprint when they encode samples the same way, and
    each still gets its own cost.

    fails if the fingerprint depends on anything but the encoding, for
    example the program's cost, or if it misses a difference between
    encodings.
    """
    a, b = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 3, 4, 16))
    evaluator = encoder(a, b)
    plain, plain_cost = evaluator.fingerprint(genome("input_0"))
    # binds `b` with itself, then leaves `a` on top, so it encodes like `plain`
    detour, detour_cost = evaluator.fingerprint(
        genome("input_1", "input_1", "hv_bind", "input_0")
    )
    other, _ = evaluator.fingerprint(genome("input_1"))
    assert plain == detour != other
    assert (plain_cost, detour_cost) == (0, 1)


def test_shift_units_looks_ahead_and_fills_the_end_with_zeros() -> None:
    """after a shift of 1, each unit contains the hv of the next unit, and the
    last unit contains zeros.

    fails if the shift wraps around and joins the end of a sequence to its
    start, if the shift goes in the wrong direction, or if it does not move
    the units.
    """
    a, b = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 2, 4, 8))
    # the bind with `b` shows where each unit moves to. a sum of `a` alone
    # only shows which units are left
    hvs, _ = encoder(a, b, unit_ops=True).encode(
        genome("input_0", 1, "hv_shift_units", "input_1", "hv_bind")
    )
    assert hvs is not None
    assert np.array_equal(hvs, (a[:, 1:] * b[:, :-1]).sum(axis=1))


def test_keep_units_keeps_the_range_between_two_ints_either_way() -> None:
    """a keep of units 1 to 2 gives the same hvs if either int is on top.

    fails if the operation reads the ints as a start and a count, or reads
    them in one fixed order. Push makes it easy to use the wrong order. also
    fails if the range does not include its last unit.
    """
    a = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 4, 8))
    evaluator = encoder(a, unit_ops=True)
    for ints in ((1, 2), (2, 1)):
        hvs, _ = evaluator.encode(genome("input_0", *ints, "hv_keep_units"))
        assert hvs is not None
        assert np.array_equal(hvs, a[:, 1:3].sum(axis=1))


def test_program_whose_hvs_are_all_zeros_scores_zero() -> None:
    """a program that sets all units to zero gets a score of 0, the same as a
    program with no output.

    fails if hvs that are all zero go to the classifier. there, the zero
    prototypes cause a division by zero, and each guess becomes the first
    class.
    """
    a = np.ones((2, 4, 8))
    evaluator = encoder(a, unit_ops=True)
    # a shift of 4 moves all 4 units out, so only zeros stay
    fitness = evaluator(genome("input_0", 4, "hv_shift_units"))
    assert fitness == Fitness(accuracy=0.0, cost=0)


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
    shared = encoder(a, b, unit_ops=True)
    copied = encoder(a, np.repeat(b, 3, axis=0), unit_ops=True)
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
    together = encoder(a, b, unit_ops=True, lengths=lengths)
    for program in PROGRAMS_FOR_LENGTHS:
        hvs, cost = together.encode(program)
        assert hvs is not None
        for i, length in enumerate(lengths):
            alone = encoder(a[i : i + 1, :length], b[:, :length], unit_ops=True)
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
    looked_up = encoder(
        Lookup(letters, symbols),
        Lookup(places, np.arange(8)[None]),
        unit_ops=True,
        lengths=lengths,
    )
    arrays = encoder(
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
    first = encoder(a, b, unit_ops=True, lengths=lengths)
    second = encoder(other, b, unit_ops=True, lengths=lengths)
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
        evaluator = encoder(a, unit_ops=True, lengths=lengths)
        assert evaluator.fingerprint(short)[0] != evaluator.fingerprint(long)[0]
