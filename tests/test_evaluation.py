"""Tests for scoring Push programs as HDC encoders, on tiny made-up data."""

import numpy as np
import pytest

from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.fitness import Fitness
from tests.helpers import evaluator_on, genome

# Tests


def test_encoding_permutes_each_hv_and_bundles_units() -> None:
    """a program runs over every unit, then each sample's units get bundled.

    fails if inputs get mixed up, if permute shifts values between samples or
    units instead of within each hv, if the wrong axis gets bundled, if bind
    costs nothing, or if permute has a cost.
    """
    a, b = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 3, 4, 16))
    hvs, cost = evaluator_on(a, b).encode(
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
    # the only class 9 sample cannot be guessed right, since its class is
    # missing from the other folds. with folds of 3, the other folds always
    # still have some 5s and 7s, so everything else gets guessed right
    labels = np.array([5, 5, 5, 5, 7, 7, 7, 7, 9])
    samples = classes[[0, 0, 0, 0, 1, 1, 1, 1, 2]]
    evaluator = Evaluator(
        [samples[:, None, :]], labels, np.arange(9), folds=3, seed=0
    )
    assert evaluator(genome("input_0")).accuracy == pytest.approx(8 / 9)


def test_folds_keep_the_samples_of_each_source_together() -> None:
    """with `sources`, each sample is held out one time, and no source has
    samples on both sides of a fold. fewer sources than folds raise an error.

    fails if the samples of a source get split over folds, if a sample is
    held out twice or never, or if the sources of test samples count.
    """
    # samples 0 and 1 are test. the sources are out of order
    sources = np.random.default_rng(0).permutation(np.repeat(np.arange(8), 4))
    samples = np.arange(2, 32)
    inputs = [np.ones((32, 1, 4))]
    evaluator = Evaluator(
        inputs, np.zeros(32), samples, folds=3, seed=0, sources=sources
    )
    held_out = np.concatenate([held for _, held in evaluator.splits])
    assert np.array_equal(np.sort(held_out), np.arange(len(samples)))
    seen = sources[samples]
    for train, held in evaluator.splits:
        assert not set(seen[train]) & set(seen[held])
    # 2 sources of the samples that evolution sees, and a third of test
    with pytest.raises(ValueError, match="fewer than the 3 folds"):
        Evaluator(
            inputs,
            np.zeros(32),
            samples,
            folds=3,
            seed=0,
            sources=np.r_[[9, 9], np.repeat([0, 1], 15)],
        )


def test_folds_without_sources_split_one_shuffle_of_the_samples() -> None:
    """without `sources`, the folds are the parts of one shuffle of the
    samples by the seed.

    fails if the folds draw other random numbers when there are no sources,
    which would change the folds, and so the scores, of every seeded run.
    """
    samples = np.arange(3, 20)
    evaluator = Evaluator(
        [np.ones((20, 1, 4))], np.zeros(20), samples, folds=3, seed=5
    )
    shuffled = np.random.default_rng(5).permutation(samples)
    # the splits give places in `samples`, and sample s is in place s - 3
    for (_, held), part in zip(
        evaluator.splits, np.array_split(shuffled, 3), strict=True
    ):
        assert np.array_equal(held, part - 3)


def test_samples_outside_evolution_get_encoded_but_not_scored() -> None:
    """samples that are not in `samples` do not count toward accuracy, and
    `encode` still gives them their own hvs.

    fails if scoring reads labels or hvs by sample number where it has places
    in `samples`, or if `encode` puts the other samples' hvs in the wrong
    places.
    """
    classes = np.random.default_rng(0).choice([-1.0, 1.0], size=(3, 64))
    # the same samples as in `test_each_sample_is_scored_without_its_own_fold`,
    # after a few samples of another class that evolution does not see
    labels = np.array([0, 0, 0, 5, 5, 5, 5, 7, 7, 7, 7, 9])
    hvs = classes[[0, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2]]
    evaluator = Evaluator(
        [hvs[:, None, :]], labels, np.arange(3, 12), folds=3, seed=0
    )
    assert evaluator(genome("input_0")).accuracy == pytest.approx(8 / 9)
    encoded, _ = evaluator.encode(genome("input_0"))
    assert encoded is not None
    assert np.array_equal(encoded, hvs)


def test_program_with_no_output_scores_zero() -> None:
    """a program that leaves no hv scores 0 and still pays for its ops.

    fails if the evaluator crashes on the missing output, or if the program's
    ops go unpaid.
    """
    a = np.ones((2, 1, 8))
    assert evaluator_on(a)(genome("hv_bind")) == Fitness(accuracy=0.0, cost=1)


def test_bundle_adds_hvs() -> None:
    """`hv_bundle` adds two hvs and keeps the sums.

    fails if bundle errors and the interpreter skips it, which leaves the last
    input as the output, or if it rounds sums back to -1s and 1s.
    """
    a, b = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 2, 3, 16))
    hvs, _ = evaluator_on(a, b).encode(
        genome("input_0", "input_1", "hv_bundle")
    )
    assert hvs is not None
    assert np.array_equal(hvs, (a + b).sum(axis=1))


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
    hvs, _ = evaluator_on(a, b, unit_ops=True).encode(
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
    evaluator = evaluator_on(a, unit_ops=True)
    for ints in ((1, 2), (2, 1)):
        hvs, _ = evaluator.encode(genome("input_0", *ints, "hv_keep_units"))
        assert hvs is not None
        assert np.array_equal(hvs, a[:, 1:3].sum(axis=1))


def test_program_whose_hvs_are_all_zeros_scores_zero() -> None:
    """a program that sets all units to zero gets a score of 0, the same as a
    program with no output.

    fails if hvs that are all zero go to the classifier. there, each guess
    becomes the first class.
    """
    a = np.ones((2, 4, 8))
    evaluator = evaluator_on(a, unit_ops=True)
    # a shift of 4 moves all 4 units out, so only zeros stay
    fitness = evaluator(genome("input_0", 4, "hv_shift_units"))
    assert fitness == Fitness(accuracy=0.0, cost=0)


def test_a_class_whose_train_hvs_are_all_zero_can_be_guessed() -> None:
    """a class whose train hvs are all zero gets a prototype that every sample
    is 0 similar to, so a sample that is less like the other classes goes to
    that class.

    fails if the zero prototype divides by zero. then every guess becomes that
    class.
    """
    hvs = np.array([[1, 1], [1, 1], [0, 0], [0, 0], [1, 1], [-1, -1]])
    labels = np.array([0, 0, 1, 1, 0, 1])
    accuracy = centroid_accuracy(hvs, labels, np.arange(4), np.array([4, 5]))
    assert accuracy == 1.0
