"""Tests for scoring Push programs as HDC encoders, on tiny made-up data."""

import pickle
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
def encoder(*inputs: np.ndarray) -> Evaluator:
    samples = np.arange(len(inputs[0]))
    return Evaluator(inputs, np.zeros(len(samples)), samples, folds=2, seed=0)


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


def test_genome_encodes_the_same_every_time() -> None:
    """bundle breaks ties at random, but a genome encodes the same way on
    every run and after pickling, like in a worker process.

    fails if the tie-breaking rng is unseeded or carries over between runs,
    which would give a genome different scores in serial and parallel runs.
    """
    a, b = np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 2, 3, 256))
    evaluator = encoder(a, b)
    bundled = genome("input_0", "input_1", "hv_bundle")
    first, _ = evaluator.encode(bundled)
    again, _ = evaluator.encode(bundled)
    in_worker, _ = pickle.loads(pickle.dumps(evaluator)).encode(bundled)
    assert first is not None
    np.testing.assert_array_equal(again, first)
    np.testing.assert_array_equal(in_worker, first)
