"""Tests for scoring Push programs as HDC encoders, on tiny made-up data."""

from typing import Any

import numpy as np
import pytest

from pushhdc.evaluation import Evaluator
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


# for tests that only encode, where labels and folds don't matter
def encoder(*inputs: np.ndarray, unit_ops: bool = False) -> Evaluator:
    samples = np.arange(len(inputs[0]))
    labels = np.zeros(len(samples))
    return Evaluator(
        inputs, labels, samples, folds=2, seed=0, unit_ops=unit_ops
    )


# Tests


def test_encoding_permutes_each_hv_and_bundles_units() -> None:
    """a program runs over every unit, then each sample's units get bundled.

    fails if inputs get mixed up, if permute shifts values between samples or
    units instead of within each hv, if the wrong axis gets bundled, or if the
    ops cost nothing.
    """
    a, b = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 3, 4, 16))
    hvs, cost = encoder(a, b).encode(
        genome("input_1", 1, "hv_permute", "input_0", "hv_bind")
    )
    assert hvs is not None
    assert np.array_equal(hvs, (a * np.roll(b, 1, axis=-1)).sum(axis=1))
    assert cost == 2


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

    fails if the fingerprint depends on anything but the encoding, like the
    program's cost, or if it misses a difference between encodings.
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
    # the bind with `b` shows which units line up. a sum of `a` alone stays
    # the same when the units move
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
    assert fitness == Fitness(accuracy=0.0, cost=1)
