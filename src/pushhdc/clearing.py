"""Keeps programs that do something new ahead of copies of better programs.

Evolution picks parents from the top ranks. Copies of the best program can
soon fill those ranks, and then a new idea is gone before it can get better.
Clearing (Petrowski 1996) moves each program that does the same thing as a
better one behind the programs that do something new.

Two programs do the same thing when they make the same pairs of samples
alike, which means they have the same kernel. Their hvs can still look
unrelated, for example when one program has an extra permute.
"""

from pushhdc.fitness import Scored
from pushhdc.kernel import Kernel


def clear(
    ranked: Scored, kernels: list[Kernel | None], similarity: float
) -> Scored:
    """`ranked` with each genome that does the same thing as a better genome
    moved behind the genomes that do something new.

    two genomes do the same thing when their kernels align at least
    `similarity`. a genome with no kernel tells no samples apart, so all of
    those do the same thing. the order inside each group stays the same, so
    the best genome stays first.
    """
    new, cleared = [], []
    kept: list[Kernel | None] = []
    for pair, kernel in zip(ranked, kernels, strict=True):
        if any(_alike(kernel, other, similarity) for other in kept):
            cleared.append(pair)
        else:
            new.append(pair)
            kept.append(kernel)
    return new + cleared


def _alike(a: Kernel | None, b: Kernel | None, similarity: float) -> bool:
    if a is None or b is None:
        return a is b
    return a.alignment(b) >= similarity
