"""Picks the program that a run gives back, using evaluators that evolution
never saw.

One evaluator makes some programs look better than they are, and evolution
favors those programs. So `choose` scores the finalists of a run again on new
evaluators and takes the mean, and `tune_ints` then tries other values for
the ints of the chosen program on the same evaluators. Both treat a gap in
the mean of at most `MEAN_TOLERANCE` as noise.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

from pushhdc.fitness import Evaluate, Fitness, Scored, ScoredGenome
from pushhdc.parallel import scoring
from pushhdc.pyshgp.push.atoms import Literal
from pushhdc.pyshgp.push.types import PushInt

# a gap in mean accuracy over many evaluators that is at most this big is
# likely noise. `choose` takes the cheapest finalist within this gap of the
# best mean, and `tune_ints` keeps a change only when it gains more than this
MEAN_TOLERANCE = 0.005

# each value on the grid is about this many times the value before it, so
# every scale gets the same number of tries
_GRID_RATIO = 2 ** (1 / 3)

# the most extra values that the second pass tries for one int, spread
# between the tried values just below and just above its best value
_FINE_COUNT = 8

# a change to a genome, given as the place of one of its ints and the new
# value for that int
type _Change = tuple[int, int]

# for each int, keyed by its place in the genome, the fitness at each value
# that it was tried at
type _Tried = dict[int, dict[int, Fitness]]


def choose(
    finalists: Scored,
    make_evaluators: Callable[[], Iterable[Evaluate]],
    *,
    workers: int = 1,
    mean_tolerance: float = MEAN_TOLERANCE,
) -> ScoredGenome:
    """the cheapest finalist whose mean accuracy over the evaluators from
    `make_evaluators` is within `mean_tolerance` of the best mean, with that
    mean as its accuracy.

    when one evaluator ranks many genomes, the genome on top is often the one
    whose score that evaluator happened to raise the most, and a new evaluator
    does not share that luck. because every evaluator here scores every
    finalist, luck that helps all finalists the same cancels out, which lets
    a small tolerance tell a real gain from noise.

    `make_evaluators` can make the evaluators one at a time, so that only one
    is in memory, and each evaluator scores in `workers` processes (see
    `pushhdc.parallel`).
    """
    genomes = [genome for genome, _ in finalists]
    means = np.mean(
        [
            [fitness.accuracy for fitness in fitnesses]
            for fitnesses in _on_each_evaluator(
                genomes, make_evaluators, workers
            )
        ],
        axis=0,
    )
    close = np.flatnonzero(means >= means.max() - mean_tolerance)
    chosen = min(close, key=lambda i: (finalists[i].fitness.cost, -means[i]))
    genome, fitness = finalists[chosen]
    return ScoredGenome(genome, Fitness(float(means[chosen]), fitness.cost))


def tune_ints(
    genome: Any,
    make_evaluators: Callable[[], Iterable[Evaluate]],
    *,
    top: int,
    workers: int = 1,
    mean_tolerance: float = MEAN_TOLERANCE,
) -> ScoredGenome:
    """tries other values from 0 to `top` for each int of `genome`, and gives
    back the genome with the one change that has the best mean accuracy over
    the evaluators from `make_evaluators`, with that mean as its accuracy, as
    `choose` does. the change is kept only when its mean beats the mean of
    `genome` by more than `mean_tolerance`. otherwise `genome` comes back
    unchanged with its own mean.

    the best of many changes often gains only from the luck of these
    evaluators and then does not help on new ones, while a change that fixes
    the program usually gains more than that luck. the default tolerance is
    the one that `choose` uses to tell a real gain from noise. the change is
    picked on these evaluators, so its accuracy here is a little too high, as
    for any pick of the best of many.

    `make_evaluators` must make the same evaluators each time that it is
    called, because tuning calls it up to two times. it can make them one at
    a time, so that only one is in memory, and each evaluator scores in
    `workers` processes (see `pushhdc.parallel`).
    """
    # evolution rarely moves an int to a better value, because new genes get
    # random ints and one small step changes accuracy by less than the noise
    # of one evaluator. so each int is first tried at 0 and at values up to
    # `top` that grow by a fixed ratio, which gives every scale the same
    # number of tries. then it is tried at more values between the values on
    # either side of its best one
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
        return ScoredGenome(genome, base)
    return ScoredGenome(_with_int(genome, *change), fitness)


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


# the untried values between the best value of an int and the tried values
# just below and just above it. when values tie on accuracy, the smallest one
# counts as the best
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
# the int that comes first in the genome wins, and then the smallest value.
# the change is none when no change beats `base`
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


# the mean accuracy of each genome over the evaluators, and its cost
def _means(
    genomes: list[Any],
    make_evaluators: Callable[[], Iterable[Evaluate]],
    workers: int,
) -> list[Fitness]:
    total, count = np.zeros(len(genomes)), 0
    for fitnesses in _on_each_evaluator(genomes, make_evaluators, workers):
        total += [fitness.accuracy for fitness in fitnesses]
        count += 1
    # the cost of a genome is the same on every evaluator, so the costs come
    # from the last one
    return [
        Fitness(float(accuracy / count), fitness.cost)
        for accuracy, fitness in zip(total, fitnesses, strict=True)
    ]


# the fitness of each genome on each evaluator from `make_evaluators`, one
# list for each evaluator. each evaluator scores every genome before the next
# one gets made
def _on_each_evaluator(
    genomes: list[Any],
    make_evaluators: Callable[[], Iterable[Evaluate]],
    workers: int,
) -> Iterator[list[Fitness]]:
    made = 0
    for evaluate in make_evaluators():
        with scoring(evaluate, workers) as score:
            fitnesses = score(genomes)
        # the next evaluator gets made only after this one is gone, so that
        # only one is in memory
        del evaluate, score
        made += 1
        yield fitnesses
    if made == 0:
        raise ValueError(
            "`make_evaluators` gave no evaluators. it must make them again "
            "each time that it is called"
        )
