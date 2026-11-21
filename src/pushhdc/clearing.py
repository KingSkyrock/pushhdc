"""Keeps programs that do something new ahead of copies of better programs.

Evolution picks parents from the top ranks. Copies of the best program can
soon fill those ranks, and then a new idea is gone before it can get better.
Clearing (Petrowski 1996) moves each program that does the same thing as a
better one behind the programs that do something new.

Two programs do the same thing when they make the same pairs of samples
alike, which means they have the same kernel. Their hvs can still look
unrelated, for example when one program has an extra permute.
"""

from __future__ import annotations

import numpy as np

from pushhdc.fitness import Ranked


class Kernel:
    """how alike a program makes each pair of samples.

    `alignment` compares two kernels, 1 for the same and about 0 for
    unrelated. it is centered kernel alignment (CKA, Kornblith et al. 2019),
    so a bias added to every hv does not count. CKA is built from HSIC, the
    Hilbert-Schmidt independence criterion. with few samples the plain
    estimate comes out too high, so this uses the unbiased HSIC of Song et
    al. (2012).

    make a kernel with `Kernel.of`, which checks that the hvs can give one.
    """

    def __init__(self, hvs: np.ndarray) -> None:
        # centering does not change the result, but it keeps a large bias
        # from making big numbers that cancel and lose precision
        x = hvs - hvs.mean(axis=0)
        gram = x @ x.T
        # the unbiased HSIC leaves out how alike each sample is with itself
        np.fill_diagonal(gram, 0)
        self._gram = gram
        self._rows = gram.sum(axis=1)
        self._total = float(self._rows.sum())
        self._scale = float(np.sqrt(max(self._hsic(self), 0.0)))

    @classmethod
    def of(cls, hvs: np.ndarray) -> Kernel | None:
        """the kernel of these hvs, one per sample. none if the hvs tell no
        samples apart, for example when they are all the same."""
        # the unbiased HSIC needs at least 4 samples
        if len(hvs) < 4:
            return None
        kernel = cls(hvs)
        return kernel if kernel._scale > 0 else None

    def alignment(self, other: Kernel) -> float:
        """how alike two kernels are, 1 for the same."""
        return self._hsic(other) / (self._scale * other._scale)

    # the unbiased HSIC of Song et al. (2012), for gram matrices with zeros on
    # the diagonal
    def _hsic(self, other: Kernel) -> float:
        n = len(self._gram)
        both = float((self._gram * other._gram).sum())
        totals = self._total * other._total / ((n - 1) * (n - 2))
        rows = 2 / (n - 2) * float(self._rows @ other._rows)
        return (both + totals - rows) / (n * (n - 3))


def clear(
    ranked: Ranked, kernels: list[Kernel | None], similarity: float
) -> Ranked:
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
