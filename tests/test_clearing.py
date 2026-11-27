"""Tests for clearing, which keeps new ideas ahead of copies of better ones."""

from pushhdc.clearing import clear
from pushhdc.fitness import Fitness, ScoredGenome
from pushhdc.kernel import Kernel
from tests.helpers import random_hvs, same_thing

# Tests


def test_clearing_moves_copies_behind_new_ideas() -> None:
    """a genome that does the same thing as a better one goes behind the
    genomes that do something new, and each group keeps its order.

    fails if copies keep their place, if the best genome moves, if the order
    inside a group changes, or if each genome with no kernel counts as new.
    """
    hvs = random_hvs(0)
    genomes = ["best", "copy of best", "new idea", "blank", "another blank"]
    kernels = [
        Kernel.of(hvs),
        Kernel.of(same_thing(hvs)),
        Kernel.of(random_hvs(1)),
        None,
        None,
    ]
    ranked = [ScoredGenome(genome, Fitness(0.9, 1)) for genome in genomes]
    order = [genome for genome, _ in clear(ranked, kernels, 0.8)]
    assert order == [
        "best",
        "new idea",
        "blank",
        "copy of best",
        "another blank",
    ]
