"""Evolutionary search over Push genomes."""

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from pushhdc.clearing import clear
from pushhdc.dedup import deduplicated
from pushhdc.fitness import (
    Evaluate,
    Fingerprints,
    Fitness,
    Kernels,
    Scored,
    ScoredGenome,
)
from pushhdc.kernel import Kernel
from pushhdc.parallel import scoring
from pushhdc.pyshgp.gp.genome import Genome
from pushhdc.pyshgp.gp.variation import Alternation, get_variation_operator

# how many genomes the run keeps at each cost for the finalists. keeping only
# the most accurate one is not enough, because one evaluator can rank a genome
# just behind a rival that is really worse
_KEPT_AT_EACH_COST = 2


@dataclass(frozen=True)
class Result:
    """the best genome of a run, its fitness, each generation's best, and the
    finalists for `choose`."""

    best_genome: Any
    best_fitness: Fitness
    history: list[Fitness]
    finalists: Scored


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
    accuracy_tolerance: float = 0.01,
    workers: int = 1,
    simplification_steps: int = 200,
    finalist_window: float = 0.02,
    niche_similarity: float | None = 0.8,
) -> Result:
    """evolves genomes toward higher accuracy and then lower cost, and gives
    back the best genome with the finalists for `choose`.

    genomes within `accuracy_tolerance` of the best accuracy so far count as
    tied with it, and cost decides between them. `history` holds the best
    fitness of each generation, starting with the random first one. at the
    end, the best genome gets simplified by deleting genes that do not help,
    so `best_fitness` can differ from the last entry of `history`.

    `finalists` are the most accurate few genomes at each cost from the whole
    run that come within `finalist_window` of the best accuracy, simplified
    like the best genome. one evaluator makes some genomes look better than
    they are, so `choose` can score them again with other evaluators.

    to keep new ideas alive as parents while they still score low, a genome
    that does the same thing as a better genome goes behind the genomes that
    do something new (see `pushhdc.clearing`). two genomes do the same thing,
    and are in the same niche, when their kernels align at least
    `niche_similarity`. this needs an `evaluate` with a `kernel` method, for
    example `Evaluator`, and `niche_similarity=None` turns it off.

    when `evaluate` also has a `fingerprint` method, genomes that share a
    fingerprint get scored once (see `pushhdc.dedup`). with `workers > 1`,
    `evaluate` runs in worker processes, so Python has to be able to copy it
    to them, which works for a function or object defined at the top level
    of a module.
    """
    rng = np.random.default_rng(seed)
    umad = get_variation_operator("umad")
    alternation = Alternation()

    def make_child(ranked: Scored) -> Any:
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
    ranker = _Ranker(
        accuracy_tolerance,
        _kernel_for_clearing(evaluate, niche_similarity),
        niche_similarity,
    )

    with scoring(evaluate, workers) as score_in_full:
        # a genome whose fingerprint was seen before gets the saved accuracy
        # and skips the full run
        score = (
            deduplicated(score_in_full, evaluate.fingerprint)
            if isinstance(evaluate, Fingerprints)
            else score_in_full
        )

        def scored(genomes: list[Any]) -> Scored:
            return [
                ScoredGenome(genome, fitness)
                for genome, fitness in zip(genomes, score(genomes), strict=True)
            ]

        ranked = ranker.rank(scored(population))
        history = [ranked[0].fitness]
        for _ in range(generations):
            children = [
                make_child(ranked) for _ in range(population_size - elites)
            ]
            ranked = ranker.rank(ranked[:elites] + scored(children))
            history.append(ranked[0].fitness)

        # a deletion that keeps the fingerprint gets the saved accuracy and
        # skips the full run
        def simplify(pair: ScoredGenome) -> ScoredGenome:
            return _simplify(
                pair,
                lambda genome: score([genome])[0],
                simplification_steps,
                rng,
            )

        best = simplify(ranked[0])
        # a shared fingerprint only shows that two genomes encode the probe
        # samples the same way, so the genome that comes back gets a full run
        # for its true score
        if score is not score_in_full:
            [fitness] = score_in_full([best.genome])
            best = ScoredGenome(best.genome, fitness)
        finalists = _finalists(
            ranker.best_at_each_cost,
            ranker.best_accuracy - finalist_window,
            simplify,
        )
    return Result(best.genome, best.fitness, history, finalists)


# the `kernel` method of `evaluate` when clearing is on. without one, `evolve`
# warns and runs without clearing
def _kernel_for_clearing(
    evaluate: Evaluate, similarity: float | None
) -> Callable[[Any], Kernel | None] | None:
    if similarity is None:
        return None
    if isinstance(evaluate, Kernels):
        return evaluate.kernel
    warnings.warn(
        "`evaluate` has no `kernel` method, so `evolve` runs without "
        "clearing. pass `niche_similarity=None` to turn clearing off.",
        stacklevel=3,
    )
    return None


# ranks each generation for `evolve`, and remembers the best accuracy and the
# best genomes at each cost of the whole run. with a `kernel` and a
# `similarity`, it clears genomes that do the same thing as a better one
class _Ranker:
    def __init__(
        self,
        tolerance: float,
        kernel: Callable[[Any], Kernel | None] | None,
        similarity: float | None,
    ) -> None:
        self.tolerance = tolerance
        self.kernel = kernel
        self.similarity = similarity
        # the tolerance counts from the best accuracy of the whole run.
        # counting from the current population would let a chain of slightly
        # worse but cheaper genomes walk accuracy down by more than the
        # tolerance
        self.best_accuracy = 0.0
        self.best_at_each_cost = _BestAtEachCost()

    def rank(self, scored: Scored) -> Scored:
        accuracies = [pair.fitness.accuracy for pair in scored]
        self.best_accuracy = max(self.best_accuracy, *accuracies)
        self.best_at_each_cost.add(scored)
        ranked = _rank(scored, self.best_accuracy - self.tolerance)
        if self.similarity is None or self.kernel is None:
            return ranked
        kernels = [self.kernel(pair.genome) for pair in ranked]
        return clear(ranked, kernels, self.similarity)


# the most accurate genomes at each cost, up to `_KEPT_AT_EACH_COST` of them.
# a genome with the same cost and accuracy as one already kept is often a
# copy of it, so it does not take a place
class _BestAtEachCost:
    def __init__(self) -> None:
        self._kept: dict[float, Scored] = {}

    def add(self, scored: Scored) -> None:
        for pair in scored:
            kept = self._kept.setdefault(pair.fitness.cost, [])
            accuracy = pair.fitness.accuracy
            if all(accuracy != other.fitness.accuracy for other in kept):
                kept.append(pair)
                kept.sort(key=lambda kept_pair: -kept_pair.fitness.accuracy)
                del kept[_KEPT_AT_EACH_COST:]

    # the kept genomes cost by cost, in the order that each cost first came
    # in, and the most accurate first within each cost
    def __iter__(self) -> Iterator[ScoredGenome]:
        return (pair for kept in self._kept.values() for pair in kept)


# the finalists for `choose`, cheapest first. every genome in
# `best_at_each_cost` that reaches `min_accuracy` gets simplified first, so
# that the genome `choose` picks is exactly the one that it scored.
# simplifying can lower the cost of a genome, so the most accurate few at
# each cost get picked again after it
def _finalists(
    best_at_each_cost: _BestAtEachCost,
    min_accuracy: float,
    simplify: Callable[[ScoredGenome], ScoredGenome],
) -> Scored:
    finalists = _BestAtEachCost()
    finalists.add(
        [
            simplify(pair)
            for pair in best_at_each_cost
            if pair.fitness.accuracy >= min_accuracy
        ]
    )
    return sorted(
        finalists,
        key=lambda pair: (pair.fitness.cost, -pair.fitness.accuracy),
    )


# makes a genome shorter and easier to read by deleting genes that do not
# help. this is the automatic simplification of Helmuth et al. (GECCO 2017).
# each try deletes a few random genes and keeps the shorter genome when it is
# at least as accurate and costs no more. pyshgp's `GenomeSimplifier` does
# the same, but it does not score genomes with `Fitness`
def _simplify(
    pair: ScoredGenome,
    evaluate: Evaluate,
    steps: int,
    rng: np.random.Generator,
) -> ScoredGenome:
    genome, fitness = pair
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
    return ScoredGenome(genome, fitness)


# sorts genomes best first, by accuracy and then cost. every accuracy of at
# least `good_enough` counts as the same, so cost decides between those genomes
def _rank(scored: Scored, good_enough: float) -> Scored:
    def key(pair: ScoredGenome) -> tuple[float, float]:
        return (-min(pair.fitness.accuracy, good_enough), pair.fitness.cost)

    return sorted(scored, key=key)


# picks `size` random genomes and returns the best of them. `ranked` is sorted
# best first, which makes the lowest picked index the winner
def _tournament(ranked: Scored, size: int, rng: np.random.Generator) -> Any:
    picks = rng.choice(len(ranked), size=min(size, len(ranked)), replace=False)
    return ranked[int(picks.min())].genome
