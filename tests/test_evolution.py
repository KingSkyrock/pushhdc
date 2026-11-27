"""Tests for the evolutionary loop, using toy evaluators instead of HDC."""

from typing import Any

import numpy as np
import pytest

from pushhdc.evolution import (
    Result,
    _BestAtEachCost,
    _simplify,
    _tournament,
    evolve,
)
from pushhdc.fitness import Evaluate, Fitness, Probe, ScoredGenome
from pushhdc.pyshgp.gp.genome import GeneSpawner
from pushhdc.pyshgp.push.atoms import InstructionMeta
from pushhdc.pyshgp.push.instruction_set import InstructionSet
from tests.helpers import genome

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


# a small seeded run. each test passes only the settings it checks.
# simplification is off unless a test turns it on, since it could shrink a
# wrongly picked best genome and hide a ranking bug. clearing is off too,
# since the toy evaluators have no kernel
def run(
    evaluate: Evaluate,
    *,
    workers: int = 1,
    max_genome_length: int = 100,
    accuracy_tolerance: float = 0.0,
    simplification_steps: int = 0,
    finalist_window: float = 0.02,
    niche_similarity: float | None = None,
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
        simplification_steps=simplification_steps,
        finalist_window=finalist_window,
        niche_similarity=niche_similarity,
    )


# makes children always lose to their parents, leaving elitism as the only way
# to keep the best. every genome scores worse than all the ones before it
class WorseEachTime:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, genome: Any) -> Fitness:
        self.calls += 1
        return Fitness(accuracy=1 / self.calls, cost=0)


def cost_only(genome: Any) -> Fitness:
    return Fitness(accuracy=1.0, cost=len(genome))


# all accuracies lie within 0.02 of each other, so under a 0.02 tolerance
# every genome ties and cost alone picks the best. accuracy still grows with
# length and crosses 0.99, so a ranking that rounded accuracy to steps of 0.02
# would put long genomes ahead of cheap ones
def nearly_tied(genome: Any) -> Fitness:
    accuracy = 0.98 + min(len(genome), 15) / 1000
    return Fitness(accuracy=accuracy, cost=len(genome))


# each genome scores 0.001 worse and costs 1 less than the one before, so new
# genomes are always cheaper and close enough to tie with recent ones
class CheaperButWorse:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, genome: Any) -> Fitness:
        self.calls += 1
        return Fitness(accuracy=1 - self.calls / 1000, cost=-self.calls)


# enough steps to finish simplifying these small genomes
def simplified(genome: Any, evaluate: Evaluate) -> ScoredGenome:
    rng = np.random.default_rng(0)
    return _simplify(ScoredGenome(genome, evaluate(genome)), evaluate, 200, rng)


# only the 7 matters for accuracy, and every gene costs 1
def needs_seven(genome: Any) -> Fitness:
    has_seven = any(gene.value == 7 for gene in genome)
    return Fitness(accuracy=1.0 if has_seven else 0.5, cost=len(genome))


# every genome is equally accurate, but shorter ones cost more
def shorter_costs_more(genome: Any) -> Fitness:
    return Fitness(accuracy=1.0, cost=-len(genome))


# scores like `toy_fitness` and counts the genomes it really scores. every
# genome gets the same fingerprint, so only the first one needs scoring
class OneFingerprint:
    def __init__(self) -> None:
        self.scored = 0

    def __call__(self, genome: Any) -> Fitness:
        self.scored += 1
        return toy_fitness(genome)

    def fingerprint(self, genome: Any) -> Probe:
        return Probe(b"same", len(genome))


# stands in for `Kernel`. genomes of the same length do the same thing
class LengthAsKernel:
    def __init__(self, length: int) -> None:
        self.length = length

    def alignment(self, other: "LengthAsKernel") -> float:
        return float(self.length == other.length)


# a toy evaluator with kernels, so that clearing can change the ranks
class ToyWithKernels:
    def __call__(self, genome: Any) -> Fitness:
        return toy_fitness(genome)

    def kernel(self, genome: Any) -> LengthAsKernel:
        return LengthAsKernel(len(genome))


# Tests


def test_same_seed_gives_same_run_serial_and_parallel() -> None:
    """a seeded run gives the same result in one process or several.

    fails if the seed does not drive every random choice, or if parallel
    evaluation scores genomes differently from serial.
    """
    assert run(toy_fitness) == run(toy_fitness, workers=2)


def test_best_fitness_never_gets_worse() -> None:
    """the best genome survives from one generation to the next.

    every child scores worse than its parents, so the first generation's best
    has to stay the best. fails if elitism is broken and the best can be lost.
    """
    history = run(WorseEachTime()).history
    assert all(fitness == history[0] for fitness in history)


def test_no_genome_is_longer_than_the_max() -> None:
    """every genome the loop evaluates fits `max_genome_length`.

    fails if children are not cut to length after crossover and mutation,
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
    ranked = [
        ScoredGenome("best", Fitness(1.0, 0)),
        ScoredGenome("worst", Fitness(0.0, 9)),
    ]
    rng = np.random.default_rng(0)
    assert _tournament(ranked, size=2, rng=rng) == "best"


def test_simplify_deletes_only_genes_that_do_not_matter() -> None:
    """simplifying leaves just the genes that accuracy depends on.

    fails if a deletion that lowers accuracy gets kept, or if deletions that
    change nothing get thrown away.
    """
    simple, fitness = simplified(genome(*range(10)), needs_seven)
    assert [gene.value for gene in simple] == [7]
    assert fitness == Fitness(accuracy=1.0, cost=1)


def test_simplify_never_raises_cost() -> None:
    """simplifying keeps a genome when every deletion would cost more.

    fails if only accuracy gets checked, which could swap a program for a
    shorter but more expensive one.
    """
    simple, _ = simplified(genome(1, 2), shorter_costs_more)
    assert len(simple) == 2


def test_evolve_returns_a_simplified_genome() -> None:
    """the best genome comes back without the genes that do not help.

    genes cost nothing, so evolution alone leaves the best genome long, and
    only simplification shrinks it to one gene. fails if `evolve` skips
    simplification or returns the genome from before it.
    """
    result = run(accuracy_only, simplification_steps=200)
    assert len(result.best_genome) == 1


def test_evolve_scores_each_fingerprint_once() -> None:
    """with an evaluator that has a `fingerprint`, only one genome per
    fingerprint gets scored, plus one full run of the genome that comes back.

    fails if `evolve` ignores the fingerprint and scores every genome, or if
    the score it returns is a saved one instead of one from a full run.
    """
    evaluator = OneFingerprint()
    result = run(evaluator)
    assert evaluator.scored == 2
    assert result.best_fitness == toy_fitness(result.best_genome)


def test_best_at_each_cost_keeps_the_most_accurate_without_copies() -> None:
    """the most accurate few genomes at each cost get kept, also at a cost
    where a cheaper genome is more accurate.

    fails if only the best genome at each cost is kept, if a copy with the
    same accuracy takes a place, or if a cost gets dropped because a cheaper
    genome beats it.
    """
    best = _BestAtEachCost()
    best.add(
        [
            ScoredGenome("a", Fitness(0.8, 2)),
            ScoredGenome("b", Fitness(0.9, 2)),
            ScoredGenome("copy of b", Fitness(0.9, 2)),
            ScoredGenome("c", Fitness(0.85, 2)),
            ScoredGenome("d", Fitness(0.7, 3)),
        ]
    )
    assert list(best) == [
        ("b", Fitness(0.9, 2)),
        ("c", Fitness(0.85, 2)),
        ("d", Fitness(0.7, 3)),
    ]


def test_finalists_are_close_to_the_best() -> None:
    """every finalist is within `finalist_window` of the most accurate genome
    the run has seen.

    fails if the window is ignored, which gives `choose` genomes that cannot
    win and wastes its evaluations.
    """
    recorder = Recorder(toy_fitness)
    finalists = run(recorder, finalist_window=0.02).finalists
    most_accurate = max(fitness.accuracy for _, fitness in recorder.seen)
    assert finalists
    assert all(f.accuracy >= most_accurate - 0.02 for _, f in finalists)


def test_evolve_returns_simplified_finalists() -> None:
    """the finalists come back without the genes that do not help.

    genes cost nothing, so only simplification shrinks a genome to one gene.
    fails if the finalists skip simplification, which lets `choose` score
    genomes that are different from the ones it returns.
    """
    finalists = run(accuracy_only, simplification_steps=200).finalists
    assert all(len(genome) == 1 for genome, _ in finalists)


def test_clearing_runs_only_when_it_is_on() -> None:
    """with kernels from `evaluate`, clearing changes the run, and
    `niche_similarity=None` turns it off.

    fails if clearing does not change the ranks, or if
    `niche_similarity=None` leaves clearing on.
    """
    on = run(ToyWithKernels(), niche_similarity=0.8)
    off = run(ToyWithKernels())
    assert on != off
    assert off == run(toy_fitness)


def test_clearing_without_a_kernel_warns() -> None:
    """`evolve` warns when clearing is on but `evaluate` has no `kernel`
    method.

    fails if `evolve` runs without clearing and gives no warning, which
    changes the search with no sign of it.
    """
    with pytest.warns(UserWarning, match="no `kernel` method"):
        run(toy_fitness, niche_similarity=0.8)
