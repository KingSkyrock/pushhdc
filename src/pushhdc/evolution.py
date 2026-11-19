"""Evolutionary search over Push genomes."""

from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

import numpy as np

from pushhdc.dedup import deduplicated
from pushhdc.pyshgp.gp.genome import Genome
from pushhdc.pyshgp.gp.variation import Alternation, get_variation_operator


@dataclass(frozen=True)
class Fitness:
    """how good a genome is. higher accuracy is better, then lower cost."""

    accuracy: float
    cost: float


type Evaluate = Callable[[Any], Fitness]
type Ranked = list[tuple[Any, Fitness]]


@dataclass(frozen=True)
class Result:
    """the best genome of a run, its fitness, each generation's best, and the
    finalists for `choose`."""

    best_genome: Any
    best_fitness: Fitness
    history: list[Fitness]
    finalists: Ranked


def evolve(
    spawner: Any,
    evaluate: Evaluate,
    *,
    seed: int | None,
    population_size: int = 100,
    generations: int = 50,
    initial_genome_size: tuple[int, int] = (10, 50),
    max_genome_length: int = 100,
    elites: int = 2,
    tournament_size: int = 7,
    crossover_rate: float = 0.5,
    accuracy_tolerance: float = 0.02,
    workers: int = 1,
    simplification_steps: int = 200,
    finalist_window: float = 0.02,
) -> Result:
    """evolve genomes toward higher accuracy, then lower cost.

    with `workers > 1`, `evaluate` runs in worker processes, so Python has to
    be able to copy it to them, which works for a function or object defined
    at the top level of a module. genomes within `accuracy_tolerance` of the
    best accuracy so far count as tied with it, and cost decides between
    them. `history` has the best fitness of each generation, starting with
    the random first one. at the end, the best genome gets simplified by
    deleting genes that do not help, so `best_fitness` can differ from the
    last entry of `history`. if `evaluate` has a `fingerprint` method, for
    example `Evaluator`, genomes that share a fingerprint get scored once
    (see `pushhdc.dedup`).

    `finalists` are the most accurate few genomes at each cost from the whole
    run that come within `finalist_window` of the best accuracy, simplified
    like the best genome. one evaluator makes some genomes look better than
    they are, so `choose` can score them again with other evaluators.
    """
    rng = np.random.default_rng(seed)
    umad = get_variation_operator("umad")
    alternation = Alternation()

    def make_child(ranked: Ranked) -> Any:
        parent = _tournament(ranked, tournament_size, rng)
        if rng.random() < crossover_rate:
            other = _tournament(ranked, tournament_size, rng)
            parent = alternation.produce([parent, other], spawner, rng=rng)
        child = umad.produce([parent], spawner, rng=rng)
        return Genome(child[:max_genome_length])

    population = [
        spawner.spawn_genome(initial_genome_size, rng=rng)
        for _ in range(population_size)
    ]
    # the tolerance counts from the best accuracy of the whole run. counting
    # from the current population would let a chain of slightly worse but
    # cheaper genomes walk accuracy down by more than the tolerance
    best_accuracy = 0.0
    best_at_each_cost: dict[float, Ranked] = {}

    def rank(scored: Ranked) -> Ranked:
        nonlocal best_accuracy
        accuracies = [fitness.accuracy for _, fitness in scored]
        best_accuracy = max(best_accuracy, *accuracies)
        _keep_best_at_each_cost(best_at_each_cost, scored)
        return _rank(scored, best_accuracy - accuracy_tolerance)

    with _evaluator(evaluate, workers) as run_all:
        fingerprint = getattr(evaluate, "fingerprint", None)
        evaluate_all = (
            run_all
            if fingerprint is None
            else deduplicated(run_all, fingerprint)
        )
        ranked = rank(
            list(zip(population, evaluate_all(population), strict=True))
        )
        history = [ranked[0][1]]
        for _ in range(generations):
            children = [
                make_child(ranked) for _ in range(population_size - elites)
            ]
            scored = list(zip(children, evaluate_all(children), strict=True))
            ranked = rank(ranked[:elites] + scored)
            history.append(ranked[0][1])

        # a deletion that keeps the fingerprint gets the saved accuracy and
        # skips the full run
        def simplify(genome: Any, fitness: Fitness) -> tuple[Any, Fitness]:
            return _simplify(
                genome,
                fitness,
                lambda genome: evaluate_all([genome])[0],
                simplification_steps,
                rng,
            )

        best_genome, best_fitness = simplify(*ranked[0])
        # a shared fingerprint does not prove the same accuracy, so the
        # returned genome gets a full run for its true score
        if fingerprint is not None:
            [best_fitness] = run_all([best_genome])

        # a finalist gets simplified before `choose` scores it, so the genome
        # that `choose` picks is the one it scored
        finalists: dict[float, Ranked] = {}
        _keep_best_at_each_cost(
            finalists,
            [
                simplify(genome, fitness)
                for level in best_at_each_cost.values()
                for genome, fitness in level
                if fitness.accuracy >= best_accuracy - finalist_window
            ],
        )
    return Result(
        best_genome,
        best_fitness,
        history,
        sorted(
            (pair for level in finalists.values() for pair in level),
            key=lambda pair: (pair[1].cost, -pair[1].accuracy),
        ),
    )


def choose(
    finalists: Ranked,
    evaluators: Iterable[Evaluate],
    *,
    mean_tolerance: float,
) -> tuple[Any, Fitness]:
    """the cheapest finalist whose mean accuracy over `evaluators` is within
    `mean_tolerance` of the best mean, with that mean as its accuracy.

    the most accurate of many genomes is often the one that got the most luck
    from its evaluator. new evaluators do not have that luck. every evaluator
    scores every finalist, so luck that helps all finalists the same cancels,
    and a small tolerance can tell a real gain from noise. `evaluators` can be
    a generator that makes one evaluator at a time, so that only one is in
    memory.
    """
    means = np.mean(
        [
            [evaluate(genome).accuracy for genome, _ in finalists]
            for evaluate in evaluators
        ],
        axis=0,
    )
    close = np.flatnonzero(means >= means.max() - mean_tolerance)
    chosen = min(close, key=lambda i: (finalists[i][1].cost, -means[i]))
    genome, fitness = finalists[chosen]
    return genome, Fitness(float(means[chosen]), fitness.cost)


# keeps the most accurate few genomes at each cost in `best`. the best one is
# not enough, because one evaluator can put a genome behind a close genome
# that is really worse. a genome with the same cost and accuracy as one in
# `best` is often a copy of it, so it does not take a place
def _keep_best_at_each_cost(best: dict[float, Ranked], scored: Ranked) -> None:
    for genome, fitness in scored:
        kept = best.setdefault(fitness.cost, [])
        if all(fitness.accuracy != other.accuracy for _, other in kept):
            kept.append((genome, fitness))
            kept.sort(key=lambda pair: -pair[1].accuracy)
            del kept[2:]


# makes a genome shorter and easier to read by deleting genes that do not
# help. this is the automatic simplification of Helmuth et al. (GECCO 2017).
# pyshgp's `GenomeSimplifier` does the same, but it does not score genomes
# with `Fitness`
def _simplify(
    genome: Any,
    fitness: Fitness,
    evaluate: Evaluate,
    steps: int,
    rng: np.random.Generator,
) -> tuple[Any, Fitness]:
    for _ in range(steps):
        if len(genome) <= 1:
            break
        count = min(int(rng.integers(1, 4)), len(genome) - 1)
        deleted = set(rng.choice(len(genome), size=count, replace=False))
        shorter = Genome(
            [gene for i, gene in enumerate(genome) if i not in deleted]
        )
        shorter_fitness = evaluate(shorter)
        if (
            shorter_fitness.accuracy >= fitness.accuracy
            and shorter_fitness.cost <= fitness.cost
        ):
            genome, fitness = shorter, shorter_fitness
    return genome, fitness


# sorts genomes best first, by accuracy and then cost. every accuracy of at
# least `good_enough` counts as the same, so cost decides between those genomes
def _rank(scored: Ranked, good_enough: float) -> Ranked:
    def key(pair: tuple[Any, Fitness]) -> tuple[float, float]:
        return (-min(pair[1].accuracy, good_enough), pair[1].cost)

    return sorted(scored, key=key)


# picks `size` random genomes and returns the best of them. `ranked` is sorted
# best first, which makes the lowest picked index the winner
def _tournament(ranked: Ranked, size: int, rng: np.random.Generator) -> Any:
    picks = rng.choice(len(ranked), size=min(size, len(ranked)), replace=False)
    return ranked[int(picks.min())][0]


_worker: dict[str, Evaluate] = {}


# keeps a dataset inside `evaluate` from being sent with every genome. it runs
# once as each worker process starts and hands it `evaluate`
def _init_worker(evaluate: Evaluate) -> None:
    _worker["evaluate"] = evaluate


def _evaluate_in_worker(genome: Any) -> Fitness:
    return _worker["evaluate"](genome)


# gives back a function that scores a list of genomes, in worker processes when
# `workers > 1`. results come back in order, which keeps parallel runs
# identical to serial ones
@contextmanager
def _evaluator(
    evaluate: Evaluate, workers: int
) -> Generator[Callable[[list[Any]], list[Fitness]]]:
    if workers <= 1:
        yield lambda genomes: [evaluate(genome) for genome in genomes]
        return
    pool = Pool(workers, initializer=_init_worker, initargs=(evaluate,))
    try:
        yield lambda genomes: list(pool.imap(_evaluate_in_worker, genomes))
    finally:
        pool.close()
        pool.join()
