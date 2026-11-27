"""How good a genome is, and what the search can ask of an evaluator.

The search (`evolve`, `choose` and `tune_ints`) and the scoring (`Evaluator`)
share only this module and the `kernel` module. An evaluator is anything that
gives a genome its `Fitness`. It can also offer `fingerprint` and `kernel`,
which `evolve` uses when they are there to skip repeated work and to keep new
ideas alive.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, runtime_checkable

from pushhdc.kernel import Kernel


@dataclass(frozen=True)
class Fitness:
    """how good a genome is. higher accuracy is better, then lower cost."""

    accuracy: float
    cost: float


class ScoredGenome(NamedTuple):
    """a genome together with its fitness."""

    genome: Any
    fitness: Fitness


# gives one genome its fitness
type Evaluate = Callable[[Any], Fitness]

# genomes with their fitness, for example one generation ranked best first
type Scored = list[ScoredGenome]


class Probe(NamedTuple):
    """what a cheap run on a few samples tells about a genome. genomes that
    encode those samples in the same way share a `fingerprint`, and `cost` is
    the cost of the genome's own program."""

    fingerprint: bytes
    cost: float


@runtime_checkable
class Fingerprints(Protocol):
    """an evaluator that can fingerprint a genome. `evolve` then gives a
    genome whose fingerprint it has seen before the accuracy that it saved
    for that fingerprint, and skips the full run (see `pushhdc.dedup`)."""

    def fingerprint(self, genome: Any) -> Probe: ...


@runtime_checkable
class Kernels(Protocol):
    """an evaluator that can tell how alike a genome makes each pair of a few
    samples. `evolve` uses these kernels to move genomes that do the same
    thing as a better genome behind the ones that do something new (see
    `pushhdc.clearing`)."""

    def kernel(self, genome: Any) -> Kernel | None: ...
