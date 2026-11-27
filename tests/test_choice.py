"""Tests for `choose` and `tune_ints`, using toy evaluators instead of HDC."""

from collections.abc import Callable
from typing import Any

from pushhdc.choice import choose, tune_ints
from pushhdc.fitness import Evaluate, Fitness, ScoredGenome
from pushhdc.pyshgp.gp.genome import Genome
from tests.helpers import genome

# Helpers

TOP = 500


def values(genome: Any) -> list[int]:
    return [gene.value for gene in genome]


# `choose` and `tune_ints` take a function that makes the evaluators
def making(*evaluators: Evaluate) -> Callable[[], list[Evaluate]]:
    return lambda: list(evaluators)


# the accuracy of each value of the first int, and 0 for other values. the
# cost is the value, so that each change has a different cost
class Table:
    def __init__(self, accuracies: dict[int, float]) -> None:
        self.accuracies = accuracies

    def __call__(self, genome: Any) -> Fitness:
        value = genome[0].value
        return Fitness(self.accuracies.get(value, 0.0), cost=value)


# an evaluator that gives each named genome a set accuracy
def scores(**accuracy: float) -> Evaluate:
    return lambda genome: Fitness(accuracy=accuracy[genome], cost=0)


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


def flat(genome: Any) -> Fitness:
    return Fitness(0.5, cost=0)


# Tests


def test_choose_takes_the_cheapest_close_to_the_best_mean() -> None:
    """`choose` averages each finalist over all evaluators, then takes the
    cheapest one within the tolerance of the best mean.

    the cheap finalist is a little behind on the mean, but far behind with
    the first evaluator. fails if `choose` trusts one evaluator, if it ignores
    cost, or if it takes the cheapest finalist when it is too far behind.
    """
    finalists = [
        ScoredGenome("cheap", Fitness(0.9, 2)),
        ScoredGenome("costly", Fitness(0.9, 3)),
    ]
    evaluators = making(
        scores(cheap=0.88, costly=0.90),
        scores(cheap=0.91, costly=0.90),
    )
    assert choose(finalists, evaluators, mean_tolerance=0.006) == (
        "cheap",
        Fitness(0.895, 2),
    )
    assert choose(finalists, evaluators, mean_tolerance=0.004) == (
        "costly",
        Fitness(0.90, 3),
    )


def test_finds_an_int_at_any_scale() -> None:
    """starting from the same int each time, tuning finds the best int
    wherever it lies between 0 and `top`, exactly when it is at most 20 and
    within 5% when it is larger, so no scale is favored. the tolerance is 0
    so that small gains close to the starting int still count.

    fails if the grid is too coarse or stops below `top`, or if the values
    between grid values get skipped or tried around the wrong value.
    """
    for target in range(TOP + 1):
        tuned, _ = tune_ints(
            genome(7), making(Peak(target)), top=TOP, mean_tolerance=0
        )
        found = values(tuned)[0]
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
        tuned, fitness = tune_ints(genome(10, 10), making(evaluate), top=TOP)
        assert values(tuned) == expected
        assert fitness.accuracy == max(gains)


def test_keeps_the_genome_when_no_change_gains() -> None:
    """a genome that no int change beats comes back unchanged with its own
    mean, and so does a genome with no ints at all.

    fails if tuning gives back a changed genome when no change gains, or if
    a genome with no ints breaks it.
    """
    for unchanged in (genome(10, 3), Genome([])):
        assert tune_ints(unchanged, making(flat, flat), top=TOP) == (
            unchanged,
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
    assert tune_ints(
        genome(10), making(equal), top=TOP, mean_tolerance=0.25
    ) == (genome(10), Fitness(0.5, cost=10))
    # these two gains sit just below and just above the default tolerance
    small = Table({10: 0.5, 3: 0.504})
    assert tune_ints(genome(10), making(small), top=TOP) == (
        genome(10),
        Fitness(0.5, cost=10),
    )
    big = Table({10: 0.5, 3: 0.506})
    assert tune_ints(genome(10), making(big), top=TOP) == (
        genome(3),
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
    tuned, fitness = tune_ints(genome(10), making(first, second), top=TOP)
    assert values(tuned) == [2]
    assert fitness == Fitness(0.6, cost=2)
