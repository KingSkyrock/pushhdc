"""Tries other values for the ints of a genome that `choose` picked."""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from pushhdc.evolution import MEAN_TOLERANCE
from pushhdc.fitness import Evaluate, Fitness
from pushhdc.parallel import scoring
from pushhdc.pyshgp.push.atoms import Literal
from pushhdc.pyshgp.push.types import PushInt

# each value on the grid is about this many times the value before it, so
# every scale gets the same number of tries
_GRID_RATIO = 2 ** (1 / 3)

# the most values that get tried around the best value of an int, between
# the tried values on each side of it
_FINE_COUNT = 8

# a change to a genome, the place of one of its ints and the new value
type _Change = tuple[int, int]

# the fitness of each value that each int got tried at, by the place of the
# int
type _Tried = dict[int, dict[int, Fitness]]


def tune_ints(
    genome: Any,
    make_evaluators: Callable[[], Iterable[Evaluate]],
    *,
    top: int,
    workers: int = 1,
    mean_tolerance: float = MEAN_TOLERANCE,
) -> tuple[Any, Fitness]:
    """the genome with the one int change that has the best mean accuracy over
    the evaluators from `make_evaluators`, with that mean as its accuracy, as
    in `choose`. if that mean is not more than `mean_tolerance` above the mean
    of `genome`, `genome` comes back with its own mean.

    the best of many changes often gains only from the luck of these
    evaluators, and then it does not help on new evaluators. a change that
    fixes the program usually gains more than that luck. the default
    tolerance is the one that `choose` uses to tell a real gain from noise.

    evolution rarely moves an int to a better value. the ints of new genes are
    random, and one small step changes accuracy less than the noise of one
    evaluator. so each int of the genome gets tried at 0 and at values up to
    `top` that grow by the same ratio, so that no scale gets more tries than
    another. then the int gets tried at more values between the tried values
    on each side of its best value.

    `make_evaluators` makes the same evaluators each time that it is called,
    one at a time, so that only one is in memory. tuning calls it up to two
    times. each evaluator scores in `workers` processes (see
    `pushhdc.parallel`). the change is picked on these evaluators, so its
    accuracy here is a little too high, as for any pick of the best of many.
    """
    places = _int_places(genome)
    grid = _grid(top)
    coarse = [
        (place, value)
        for place in places
        for value in grid
        if value != genome[place].value
    ]
    base, *scores = _means(
        [genome, *[_with_int(genome, *change) for change in coarse]],
        make_evaluators,
        workers,
    )
    # the old value of each int has the fitness of the genome
    tried = {place: {genome[place].value: base} for place in places}
    _record(tried, coarse, scores)
    fine = [
        (place, value) for place in places for value in _near_best(tried[place])
    ]
    if fine:
        changed = [_with_int(genome, *change) for change in fine]
        _record(tried, fine, _means(changed, make_evaluators, workers))

    change, fitness = _best_change(tried, base)
    if change is None or fitness.accuracy - base.accuracy <= mean_tolerance:
        return genome, base
    return _with_int(genome, *change), fitness


# 0, then whole numbers that grow by about `_GRID_RATIO` each step, up to
# `top`, and `top` too
def _grid(top: int) -> list[int]:
    values = {0, top}
    value = 1.0
    while value < top:
        values.add(round(value))
        value *= _GRID_RATIO
    return sorted(v for v in values if v <= top)


# up to `count` whole numbers between `low` and `high`, spread evenly. `low`
# and `high` are not in them
def _between(low: int, high: int, count: int = _FINE_COUNT) -> list[int]:
    inside = list(range(low + 1, high))
    if len(inside) <= count:
        return inside
    picks = np.linspace(0, len(inside) - 1, count).round().astype(int)
    return sorted({inside[i] for i in picks})


# the values between the tried values on each side of the best value of an
# int, that did not get tried yet. on a tie, the smallest value is the best
def _near_best(fitnesses: dict[int, Fitness]) -> list[int]:
    values = sorted(fitnesses)
    i = int(np.argmax([fitnesses[value].accuracy for value in values]))
    low, high = values[max(i - 1, 0)], values[min(i + 1, len(values) - 1)]
    return [v for v in _between(low, high) if v not in fitnesses]


# puts the fitness of each change into `tried`
def _record(
    tried: _Tried, changes: list[_Change], fitnesses: list[Fitness]
) -> None:
    for (place, value), fitness in zip(changes, fitnesses, strict=True):
        tried[place][value] = fitness


# the change with the best accuracy, and its fitness. on a tie, the change to
# the int that comes first in the genome wins, and then the smallest value. no
# change if none beats `base`
def _best_change(
    tried: _Tried, base: Fitness
) -> tuple[_Change | None, Fitness]:
    best: _Change | None = None
    best_fitness = base
    for place, fitnesses in tried.items():
        for value in sorted(fitnesses):
            if fitnesses[value].accuracy > best_fitness.accuracy:
                best, best_fitness = (place, value), fitnesses[value]
    return best, best_fitness


def _int_places(genome: Any) -> list[int]:
    return [
        i
        for i, gene in enumerate(genome)
        if isinstance(gene, Literal) and gene.push_type == PushInt
    ]


def _with_int(genome: Any, place: int, value: int) -> Any:
    return genome.set(place, Literal(value=value, push_type=PushInt))


# the mean accuracy of each genome over the evaluators, and its cost. each
# evaluator scores all genomes before the next one gets made
def _means(
    genomes: list[Any],
    make_evaluators: Callable[[], Iterable[Evaluate]],
    workers: int,
) -> list[Fitness]:
    total, count = np.zeros(len(genomes)), 0
    for evaluate in make_evaluators():
        with scoring(evaluate, workers) as run_all:
            fitnesses = run_all(genomes)
        total += [fitness.accuracy for fitness in fitnesses]
        count += 1
        # the next evaluator gets made only after this one is gone, so that
        # only one is in memory
        del evaluate, run_all
    if count == 0:
        raise ValueError(
            "`make_evaluators` gave no evaluators. it must make them again "
            "each time that it is called"
        )
    # the cost of a genome is the same on every evaluator, so the costs come
    # from the last one
    return [
        Fitness(float(accuracy / count), fitness.cost)
        for accuracy, fitness in zip(total, fitnesses, strict=True)
    ]
