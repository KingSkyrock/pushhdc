"""Scores each distinct encoding once, so repeated programs skip the full run.

Many evolved programs encode samples exactly like one already scored, so they
would get the same accuracy. A fingerprint from running a program on a few
samples spots those repeats at a small part of the cost of a full run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pushhdc.evolution import Fitness

type Fingerprint = Callable[[Any], tuple[bytes, float]]


def deduplicated(
    evaluate_all: Callable[[list[Any]], list[Fitness]],
    fingerprint: Fingerprint,
) -> Callable[[list[Any]], list[Fitness]]:
    """scores a list of genomes like `evaluate_all`, but only runs the first
    genome with each new fingerprint.

    a repeat gets the accuracy saved for its fingerprint and the cost of its
    own program, so a cheaper repeat still ranks ahead.
    """
    saved: dict[bytes, Fitness] = {}

    def score(genomes: list[Any]) -> list[Fitness]:
        fingerprints = [fingerprint(genome) for genome in genomes]
        new: dict[bytes, Any] = {}
        for genome, (key, _) in zip(genomes, fingerprints, strict=True):
            if key not in saved:
                new.setdefault(key, genome)
        saved.update(zip(new, evaluate_all(list(new.values())), strict=True))
        return [replace(saved[key], cost=cost) for key, cost in fingerprints]

    return score
