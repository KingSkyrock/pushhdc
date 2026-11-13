"""Tests for scoring each distinct encoding once, using toy genomes."""

from pushhdc.dedup import deduplicated
from pushhdc.evolution import Fitness

# Helpers


# runs a list of genomes and remembers which ones it ran. genomes are strings.
# ones that start with "a" score 0.9, the rest 0.5, and each costs its length
class Runner:
    def __init__(self) -> None:
        self.ran: list[str] = []

    def __call__(self, genomes: list[str]) -> list[Fitness]:
        self.ran.extend(genomes)
        return [
            Fitness(0.9 if genome[0] == "a" else 0.5, len(genome))
            for genome in genomes
        ]


# genomes that start with the same letter share a fingerprint
def by_first_letter(genome: str) -> tuple[bytes, float]:
    return genome[0].encode(), len(genome)


# Tests


def test_repeats_get_the_saved_accuracy_and_their_own_cost() -> None:
    """only the first genome with each fingerprint runs, whether its repeats
    come in the same list or a later one. the repeats get its accuracy and
    their own cost.

    fails if repeats still run, or if they get the first genome's cost, which
    would hide that a repeat is cheaper.
    """
    runner = Runner()
    score = deduplicated(runner, by_first_letter)
    first = score(["aa", "b", "a"])
    later = score(["aaa"])
    assert runner.ran == ["aa", "b"]
    assert first == [Fitness(0.9, 2), Fitness(0.5, 1), Fitness(0.9, 1)]
    assert later == [Fitness(0.9, 3)]
