"""Tests for scoring many genomes at a time in worker processes."""

import time

from pushhdc.fitness import Fitness
from pushhdc.parallel import scoring

# Helpers


# makes genome 0 finish last in parallel. the "genomes" are plain ints
def slow_first(genome: int) -> Fitness:
    if genome == 0:
        time.sleep(0.3)
    return Fitness(accuracy=genome, cost=0)


# Tests


def test_parallel_scores_line_up_with_their_genomes() -> None:
    """parallel evaluation returns each score in its genome's position.

    genome 0 is slow, so it finishes after the others. fails if results are
    taken in the order workers finish (`imap_unordered`), which pairs
    genomes with the wrong scores.
    """
    genomes = list(range(6))
    with scoring(slow_first, workers=2) as evaluate_all:
        assert evaluate_all(genomes) == [Fitness(g, 0) for g in genomes]
