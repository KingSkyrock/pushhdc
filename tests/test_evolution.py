"""Tests for the evolutionary loop, using toy evaluators instead of HDC."""

import time
from typing import Any

import numpy as np

from pushhdc.evolution import (
    Evaluate,
    Fitness,
    Result,
    _evaluator,
    _tournament,
    evolve,
)
from pushhdc.pyshgp.gp.genome import GeneSpawner
from pushhdc.pyshgp.push.atoms import InstructionMeta
from pushhdc.pyshgp.push.instruction_set import InstructionSet

# Helpers


# stands in for an hdc evaluator with something cheap and deterministic.
# "accuracy" is the share of genes that are instructions
def instruction_share(genome: Any) -> float:
    instructions = sum(isinstance(gene, InstructionMeta) for gene in genome)
    return instructions / max(len(genome), 1)


def toy_fitness(genome: Any) -> Fitness:
    return Fitness(accuracy=instruction_share(genome), cost=len(genome))


def accuracy_only(genome: Any) -> Fitness:
    return Fitness(accuracy=instruction_share(genome), cost=0)


def cost_only(genome: Any) -> Fitness:
    return Fitness(accuracy=1.0, cost=len(genome))


# makes every genome tie on accuracy under a 0.02 tolerance, leaving cost to
# pick the best. accuracy grows with length and crosses 0.99, so rounding to
# steps of 0.02 would put long genomes ahead of cheap ones
def nearly_tied(genome: Any) -> Fitness:
    accuracy = 0.98 + min(len(genome), 15) / 1000
    return Fitness(accuracy=accuracy, cost=len(genome))


# lets a test look at everything the loop evaluated, by remembering each
# genome's length and fitness. only works in a single process
class Recorder:
    def __init__(self, evaluate: Evaluate) -> None:
        self.evaluate = evaluate
        self.seen: list[tuple[int, Fitness]] = []

    def __call__(self, genome: Any) -> Fitness:
        fitness = self.evaluate(genome)
        self.seen.append((len(genome), fitness))
        return fitness


# makes children always lose to their parents, leaving elitism as the only way
# to keep the best. every genome scores worse than all the ones before it
class WorseEachTime:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, genome: Any) -> Fitness:
        self.calls += 1
        return Fitness(accuracy=1 / self.calls, cost=0)


# each genome scores 0.001 worse and costs 1 less than the one before, so new
# genomes are always cheaper and close enough to tie with recent ones
class CheaperButWorse:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, genome: Any) -> Fitness:
        self.calls += 1
        return Fitness(accuracy=1 - self.calls / 1000, cost=-self.calls)


# makes genome 0 finish last in parallel. the "genomes" are plain ints
def slow_first(genome: int) -> Fitness:
    if genome == 0:
        time.sleep(0.3)
    return Fitness(accuracy=genome, cost=0)


# a small seeded run. each test passes only the settings it checks
def run(
    evaluate: Evaluate,
    *,
    workers: int = 1,
    max_genome_length: int = 100,
    accuracy_tolerance: float = 0.0,
) -> Result:
    spawner = GeneSpawner(
        n_inputs=1,
        instruction_set=InstructionSet().register_core_by_stack({"int"}),
        literals=[1, 2],
        erc_generators=[],
    )
    return evolve(
        spawner,
        evaluate,
        seed=0,
        population_size=20,
        generations=5,
        initial_genome_size=(5, 15),
        max_genome_length=max_genome_length,
        accuracy_tolerance=accuracy_tolerance,
        workers=workers,
    )


# Tests


def test_same_seed_gives_same_run_serial_and_parallel() -> None:
    """a seeded run gives the same result in one process or several.

    fails if the seed doesn't drive every random choice, or if parallel
    evaluation scores genomes differently from serial.
    """
    assert run(toy_fitness) == run(toy_fitness, workers=2)


def test_parallel_scores_line_up_with_their_genomes() -> None:
    """parallel evaluation returns each score in its genome's position.

    genome 0 is slow, so it finishes after the others. fails if results are
    taken in the order workers finish (`imap_unordered`), which pairs
    genomes with the wrong scores.
    """
    genomes = list(range(6))
    with _evaluator(slow_first, workers=2) as evaluate_all:
        assert evaluate_all(genomes) == [Fitness(g, 0) for g in genomes]


def test_best_fitness_never_gets_worse() -> None:
    """the best genome survives from one generation to the next.

    every child scores worse than its parents, so the first generation's best
    has to stay the best. fails if elitism is broken and the best can be lost.
    """
    history = run(WorseEachTime()).history
    assert all(fitness == history[0] for fitness in history)


def test_no_genome_is_longer_than_the_max() -> None:
    """every genome the loop evaluates fits `max_genome_length`.

    fails if children aren't cut to length after crossover and mutation,
    which lets genomes grow without bound.
    """
    recorder = Recorder(toy_fitness)
    run(recorder, max_genome_length=15)
    assert max(length for length, _ in recorder.seen) <= 15


def test_ranking_prefers_accuracy_then_cost() -> None:
    """the best genome has the highest accuracy, then the lowest cost.

    with `accuracy_tolerance`, cost also decides between nearly equal
    accuracies. fails if either order is flipped, or if the tolerance is
    ignored or rounds accuracy to fixed steps.
    """
    by_accuracy = Recorder(accuracy_only)
    best = run(by_accuracy).best_fitness
    assert best.accuracy == max(
        fitness.accuracy for _, fitness in by_accuracy.seen
    )

    by_cost = Recorder(cost_only)
    best = run(by_cost).best_fitness
    assert best.cost == min(fitness.cost for _, fitness in by_cost.seen)

    within_tolerance = Recorder(nearly_tied)
    best = run(within_tolerance, accuracy_tolerance=0.02).best_fitness
    assert best.cost == min(
        fitness.cost for _, fitness in within_tolerance.seen
    )


def test_tolerance_counts_from_the_best_of_the_run() -> None:
    """the best genome stays within `accuracy_tolerance` of the most accurate
    genome the run has seen.

    fails if the tolerance counts from the current population, which lets
    accuracy walk down a little every generation.
    """
    recorder = Recorder(CheaperButWorse())
    best = run(recorder, accuracy_tolerance=0.02).best_fitness
    most_accurate = max(fitness.accuracy for _, fitness in recorder.seen)
    assert best.accuracy >= most_accurate - 0.02


def test_tournament_over_everyone_picks_the_best() -> None:
    """a tournament that includes every genome returns the best one.

    fails if selection picks the worse genome, which no other test catches
    because elitism still keeps the best.
    """
    ranked = [("best", Fitness(1.0, 0)), ("worst", Fitness(0.0, 9))]
    rng = np.random.default_rng(0)
    assert _tournament(ranked, size=2, rng=rng) == "best"
