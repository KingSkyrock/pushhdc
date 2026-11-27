"""Tests for `tune_ints`, using toy evaluators of int genes instead of HDC."""

from collections.abc import Callable
from typing import Any

from pushhdc.fitness import Evaluate, Fitness
from pushhdc.pyshgp.gp.genome import Genome
from pushhdc.pyshgp.push.atoms import Literal
from pushhdc.pyshgp.push.types import PushInt
from pushhdc.tuning import tune_ints

# Helpers

TOP = 500


def ints(*values: int) -> Genome:
    return Genome([Literal(value=v, push_type=PushInt) for v in values])


def values(genome: Any) -> list[int]:
    return [gene.value for gene in genome]


# `tune_ints` takes a function that makes the evaluators
def making(*evaluators: Evaluate) -> Callable[[], list[Evaluate]]:
    return lambda: list(evaluators)


# accuracy falls as the first int gets farther from `target`
class Peak:
    def __init__(self, target: int) -> None:
        self.target = target

    def __call__(self, genome: Any) -> Fitness:
        return Fitness(1 - abs(genome[0].value - self.target) / TOP, cost=0)


# int i of a genome adds `gains[i]` to the accuracy when it is `wanted[i]`
class EachInt:
    def __init__(self, wanted: tuple[int, ...], gains: tuple[float, ...]):
        self.wanted = wanted
        self.gains = gains

    def __call__(self, genome: Any) -> Fitness:
        accuracy = sum(
            gain
            for gene, wanted, gain in zip(
                genome, self.wanted, self.gains, strict=True
            )
            if gene.value == wanted
        )
        return Fitness(accuracy, cost=0)


# the accuracy of each value of the first int, and 0 for other values. the
# cost is the value, so that each change has a different cost
class Table:
    def __init__(self, accuracies: dict[int, float]) -> None:
        self.accuracies = accuracies

    def __call__(self, genome: Any) -> Fitness:
        value = genome[0].value
        return Fitness(self.accuracies.get(value, 0.0), cost=value)


def flat(genome: Any) -> Fitness:
    return Fitness(0.5, cost=0)


# Tests


def test_finds_an_int_at_any_scale() -> None:
    """tuning finds every best int from 0 to `top` from the same start,
    exactly up to 20 and within 5% above. so a best int of 20 or 500 gets
    found as well as 1, and no scale is favored. the tolerance is 0, so that
    the small gains near the start count too.

    fails if the grid is too coarse or stops below `top`, or if the values
    between grid values get skipped or tried around the wrong value.
    """
    for target in range(TOP + 1):
        genome, _ = tune_ints(
            ints(7), making(Peak(target)), top=TOP, mean_tolerance=0
        )
        found = values(genome)[0]
        if target <= 20:
            assert found == target
        else:
            assert abs(found - target) <= 0.05 * target


def test_changes_only_the_int_with_the_best_gain() -> None:
    """when two ints can both gain, only the int with the bigger gain changes,
    whichever of the two it is.

    fails if every gain is kept, or if the best change so far starts again at
    each int.
    """
    for gains, expected in (((0.2, 0.1), [0, 10]), ((0.1, 0.2), [10, 1])):
        evaluate = EachInt((0, 1), gains)
        genome, fitness = tune_ints(ints(10, 10), making(evaluate), top=TOP)
        assert values(genome) == expected
        assert fitness.accuracy == max(gains)


def test_keeps_the_genome_when_no_change_gains() -> None:
    """a genome that no int change beats comes back as it is, with its own
    mean. a genome with no ints too.

    fails if a genome with no ints breaks tuning.
    """
    for genome in (ints(10, 3), Genome([])):
        assert tune_ints(genome, making(flat, flat), top=TOP) == (
            genome,
            Fitness(0.5, cost=0),
        )


def test_keeps_the_genome_when_the_gain_is_in_the_tolerance() -> None:
    """a change is kept only when its mean is more than the tolerance above
    the mean of the genome. a gain equal to the tolerance is not kept. the
    mean of the genome is not 0 here, so a gain counted from 0 fails.

    fails if every gain is kept, if a gain equal to the tolerance is kept, if
    the gain is not counted from the mean of the genome, if the genome comes
    back with the mean of the change, or if the default tolerance is not near
    the tolerance of `choose`.
    """
    # these accuracies and their difference are exact in floats
    equal = Table({10: 0.5, 3: 0.75})
    assert tune_ints(ints(10), making(equal), top=TOP, mean_tolerance=0.25) == (
        ints(10),
        Fitness(0.5, cost=10),
    )
    small = Table({10: 0.5, 3: 0.504})
    assert tune_ints(ints(10), making(small), top=TOP) == (
        ints(10),
        Fitness(0.5, cost=10),
    )
    big = Table({10: 0.5, 3: 0.506})
    assert tune_ints(ints(10), making(big), top=TOP) == (
        ints(3),
        Fitness(0.506, cost=3),
    )


def test_the_mean_over_evaluators_decides() -> None:
    """the change with the best mean over all evaluators wins, with that mean
    as its accuracy and its own cost, also when each evaluator alone picks
    another change.

    fails if one evaluator decides, if the accuracy is a sum or the score of
    one evaluator, or if the cost is not the cost of the genome that wins.
    """
    first = Table({0: 1.0, 1: 0.0, 2: 0.6})
    second = Table({0: 0.0, 1: 1.0, 2: 0.6})
    genome, fitness = tune_ints(ints(10), making(first, second), top=TOP)
    assert values(genome) == [2]
    assert fitness == Fitness(0.6, cost=2)
