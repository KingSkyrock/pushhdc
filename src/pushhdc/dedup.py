"""Scores each distinct encoding once, so repeated programs skip the full run.

Many evolved programs encode samples exactly like one already scored, so they
would get the same accuracy. A fingerprint from running a program on a few
samples spots those repeats at a small part of the cost of a full run.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from pushhdc.fitness import Fitness, Probe


def deduplicated(
    score_in_full: Callable[[list[Any]], list[Fitness]],
    fingerprint: Callable[[Any], Probe],
) -> Callable[[list[Any]], list[Fitness]]:
    """scores a list of genomes like `score_in_full`, but only runs the first
    genome with each new fingerprint.

    a repeat gets the accuracy saved for its fingerprint and the cost of its
    own program, so a cheaper repeat still ranks ahead.
    """
    saved: dict[bytes, Fitness] = {}

    def score(genomes: list[Any]) -> list[Fitness]:
        probes = [fingerprint(genome) for genome in genomes]
        new: dict[bytes, Any] = {}
        for genome, probe in zip(genomes, probes, strict=True):
            if probe.fingerprint not in saved:
                new.setdefault(probe.fingerprint, genome)
        saved.update(zip(new, score_in_full(list(new.values())), strict=True))
        return [
            replace(saved[probe.fingerprint], cost=probe.cost)
            for probe in probes
        ]

    return score
