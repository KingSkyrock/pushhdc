"""Evolutionary search over Push genomes."""

from collections.abc import Callable, Generator
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


@dataclass(frozen=True)
class Result:
    """the best genome of a run, its fitness, and each generation's best."""

    best_genome: Any
    best_fitness: Fitness
    history: list[Fitness]


type Evaluate = Callable[[Any], Fitness]
type Ranked = list[tuple[Any, Fitness]]


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
) -> Result:
    """evolve genomes toward higher accuracy, then lower cost.

    with `workers > 1`, `evaluate` runs in worker processes, so Python has to
    be able to copy it to them, which works for a function or object defined
    at the top level of a module. genomes within `accuracy_tolerance` of the
    best accuracy so far count as tied with it, and cost decides between
    them. `history` has the best fitness of each generation, starting with
    the random first one. at the end, the best genome gets simplified by
    deleting genes that don't help, so `best_fitness` can beat the last entry
    of `history`. if `evaluate` has a `fingerprint` method, like `Evaluator`
    does, genomes that share a fingerprint get scored once (see
    `pushhdc.dedup`).
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

    def rank(scored: Ranked) -> Ranked:
        nonlocal best_accuracy
        accuracies = [fitness.accuracy for _, fitness in scored]
        best_accuracy = max(best_accuracy, *accuracies)
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

        # a deletion that leaves the probes' encoding as it was hits the cache
        # and skips the full run
        best_genome, best_fitness = _simplify(
            *ranked[0],
            lambda genome: evaluate_all([genome])[0],
            simplification_steps,
            rng,
        )
        # the cache trusts a few probe samples, so the genome we return gets a
        # full run to make sure its score is real
        if fingerprint is not None:
            [best_fitness] = run_all([best_genome])
    return Result(best_genome, best_fitness, history)


# makes a genome shorter and easier to read by deleting genes that don't help.
# it's the automatic simplification of Helmuth et al. (GECCO 2017). pyshgp's
# `GenomeSimplifier` does the same, but scores genomes its own way instead of
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
