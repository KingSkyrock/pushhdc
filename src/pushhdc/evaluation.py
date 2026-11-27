"""Scores Push programs as HDC encoders, for `evolve` to rank."""

import hashlib
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from pushhdc.encoder_ops import (
    encoder_instructions,
    encoder_spawner,
    largest_int,
)
from pushhdc.fitness import Fitness, Probe
from pushhdc.inputs import Lookup, SameLength, by_length, checked_lengths
from pushhdc.kernel import Kernel
from pushhdc.pyshgp.gp.genome import GeneSpawner, genome_to_code
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.recording import record_ops, replay

# the evaluator keeps the probe results of this many recent genomes, which
# covers a few generations of a usual population
_KEPT_PROBE_RESULTS = 1024

# how many samples `fingerprint` and `kernel` run on
_PROBE_COUNT = 64


# what `Evaluator` keeps for each genome that it ran on the probe samples
class _ProbeResult(NamedTuple):
    fingerprint: bytes
    cost: int
    kernel: Kernel | None


class Evaluator:
    """scores a genome as an hv encoder by its cross-validated accuracy and
    its cost, which is what `evolve` ranks genomes by.

    each input is a batch of hvs shaped (samples, units, dimension), where a
    unit is one piece of a sample, such as one position of a sequence. a
    program applies the same ops to every unit, so even a short program
    encodes every piece in the same way, and the evaluator then bundles the
    units of each sample into a single hv. with `unit_ops`, a program can also
    combine each unit with a later one and keep only some of the units, which
    lets evolution decide how the units group together. this only makes sense
    when the units are in order, as the positions of a sequence are.

    `samples` are the samples that evolution may see, which is everything but
    the test split. k-fold cross-validation scores each of the `folds` parts
    with prototypes built from the other parts, so every sample counts toward
    accuracy and none is judged by a prototype that it helped build.

    `sources` gives the source of each sample, the test samples included, for
    example the person that a recording came from. each part then holds all
    the samples of its sources. samples from one source are often more alike
    than other samples, so if a source had samples on both sides of a fold, a
    program that tells sources apart would score too high.

    an input that is the same for every sample, for example the hv of each
    position, can have 1 in place of the number of samples. it then takes
    less memory and makes programs faster, and the scores stay the same.

    an input can also be a `Lookup`, which holds one hv for each symbol and
    the symbol at each unit. the padded array then never gets made, and the
    scores are the same as with it. the hvs of the units are made only while a
    program reads them, so an evaluator with many such inputs stays small,
    also in worker processes.

    `lengths` gives the number of units of each sample in `inputs`, the test
    samples included. sample i then has only its first `lengths[i]` units in
    each input, the shared inputs included. the units after that are left
    out, so they can hold any values, and each sample gets the same hv that it
    would get alone. the program runs one time for each different length, so
    many different lengths make scoring slower.
    """

    def __init__(
        self,
        inputs: Sequence[np.ndarray | Lookup],
        labels: np.ndarray,
        samples: np.ndarray,
        *,
        folds: int = 5,
        seed: int,
        unit_ops: bool = False,
        lengths: Sequence[int] | np.ndarray | None = None,
        sources: Sequence[Any] | np.ndarray | None = None,
    ) -> None:
        self.arity = len(inputs)
        lengths = checked_lengths(inputs, lengths)
        # the ints of the unit operations come from the lengths of the samples
        # that evolution sees, never from the test samples
        self.seen_lengths = lengths[samples]
        # the largest int that new genes get with `unit_ops`
        self.largest_int = largest_int(self.seen_lengths)
        # the samples get shuffled in case they are sorted by class, which
        # could put most of a class in one fold. this happens once, so every
        # genome gets the same folds. with `sources`, the folds come from a
        # shuffle of the sources instead
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(samples)
        held_out = (
            np.array_split(shuffled, folds)
            if sources is None
            else _held_out_by_source(samples, sources, folds, len(lengths), rng)
        )
        self.splits = _folds(samples, held_out, len(lengths))
        self.seen_labels = labels[samples]
        # the samples outside `samples`, such as test, are only encoded at
        # the end of a run, so they are grouped on their own and scoring never
        # runs programs on them
        self.seen_samples = samples
        self.other_samples = np.setdiff1d(np.arange(len(lengths)), samples)
        self.seen_by_length = by_length(inputs, lengths, samples)
        self.other_by_length = by_length(inputs, lengths, self.other_samples)
        self.probes_by_length = by_length(
            inputs, lengths, _probe_samples(shuffled, samples, lengths)
        )
        self._probe_results: dict[Any, _ProbeResult] = {}
        self.unit_ops = unit_ops
        self.instructions = encoder_instructions(unit_ops)

    def spawner(self) -> GeneSpawner:
        """makes random genes that match this evaluator's inputs and ops."""
        return encoder_spawner(
            self.instructions,
            self.arity,
            unit_ops=self.unit_ops,
            seen_lengths=self.seen_lengths,
        )

    def encode(self, genome: Any) -> tuple[np.ndarray | None, int]:
        """encodes every sample, the ones outside `samples` included, and
        gives back their hvs shaped (samples, dimension) with the program's
        cost. it is meant for the end of a run, to score other splits such as
        test. the hvs are none when the program leaves nothing on the
        hypervector stack."""
        seen, cost = self._run(genome, self.seen_by_length)
        if seen is None:
            return None, cost
        count = len(self.seen_samples) + len(self.other_samples)
        hvs = np.empty((count, *seen.shape[1:]), seen.dtype)
        hvs[self.seen_samples] = seen
        # a program that leaves an hv for some samples leaves one for all of
        # them, because no op chooses what to do by the values of an hv
        if len(self.other_samples):
            hvs[self.other_samples], _ = self._run(genome, self.other_by_length)
        return hvs, cost

    def fingerprint(self, genome: Any) -> Probe:
        """a fingerprint of how the program encodes a few probe samples, with
        the program's cost. programs that encode the probes in the same way
        share a fingerprint, and `evolve` runs only the first of them in full
        and gives the others its accuracy. programs that leave no hv all
        score 0, so they share an empty fingerprint."""
        result = self._probe(genome)
        return Probe(result.fingerprint, result.cost)

    def kernel(self, genome: Any) -> Kernel | None:
        """how alike the program makes each pair of probe samples, which
        `evolve` uses to tell programs apart by what they do. it is none when
        the program tells no probes apart."""
        return self._probe(genome).kernel

    # `evolve` asks for the fingerprint and then the kernel of each new
    # genome, and both come from the same run on the probe samples. so
    # `_probe` keeps the results of the most recent genomes and drops the
    # oldest ones, which keeps the memory small
    def _probe(self, genome: Any) -> _ProbeResult:
        if genome not in self._probe_results:
            hvs, cost = self._run(genome, self.probes_by_length)
            if len(self._probe_results) >= _KEPT_PROBE_RESULTS:
                del self._probe_results[next(iter(self._probe_results))]
            self._probe_results[genome] = (
                _ProbeResult(b"", cost, None)
                if hvs is None
                else _ProbeResult(
                    hashlib.sha256(hvs.tobytes()).digest(), cost, Kernel.of(hvs)
                )
            )
        return self._probe_results[genome]

    # the kept probe results only help the process that made them, so the
    # copy of the evaluator that a worker process gets leaves them out
    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_probe_results": {}}

    # runs the program on each group of samples of one length and gives back
    # one hv for each sample. scoring, `encode`, `fingerprint` and `kernel`
    # all use it, each on its own samples, and the hvs come back in the order
    # of the `rows` that `by_length` made the groups from
    def _run(
        self, genome: Any, groups: list[SameLength]
    ) -> tuple[np.ndarray | None, int]:
        signature = ProgramSignature(
            arity=self.arity, output_stacks=["hypervector"]
        )
        program = Program(code=genome_to_code(genome), signature=signature)
        # every length does the same ops, so the interpreter runs the program
        # once and numpy repeats its ops on each length (see
        # `pushhdc.recording`)
        recording, cost = record_ops(program, self.instructions)
        if recording is None:
            return None, cost
        sums = [
            replay(recording, group.inputs).data.sum(axis=1) for group in groups
        ]
        count = sum(len(group.places) for group in groups)
        hvs = np.empty((count, sums[0].shape[1]), sums[0].dtype)
        for group, group_sums in zip(groups, sums, strict=True):
            # a program that only uses inputs shared by all samples gives one
            # hv for the group, which is then the hv of every sample in it
            hvs[group.places] = group_sums
        return hvs, cost

    def __call__(self, genome: Any) -> Fitness:
        hvs, cost = self._run(genome, self.seen_by_length)
        # hvs that are all zero contain no information
        if hvs is None or not hvs.any():
            return Fitness(accuracy=0.0, cost=cost)
        accuracies = [
            centroid_accuracy(hvs, self.seen_labels, train, held_out)
            for train, held_out in self.splits
        ]
        # weighting by fold size gives the share of all samples guessed right
        weights = [len(held_out) for _, held_out in self.splits]
        return Fitness(float(np.average(accuracies, weights=weights)), cost)


def centroid_accuracy(
    hvs: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray
) -> float:
    """accuracy on `test` of a nearest-centroid classifier, which has nothing
    to tune. each class's prototype is the sum of its `train` hvs, and a
    sample goes to the most similar prototype by cosine similarity."""
    classes = np.unique(labels[train])
    prototypes = np.stack(
        [hvs[train][labels[train] == label].sum(axis=0) for label in classes]
    ).astype(float)
    # only the prototypes get normalized, because the norm of a sample is the
    # same against every class and so cannot change which class is most
    # similar. a class whose train hvs are all zero has no direction, so its
    # prototype stays zero and every sample is 0 similar to it
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes /= np.where(norms == 0, 1, norms)
    guesses = classes[(hvs[test] @ prototypes.T).argmax(axis=1)]
    return float((guesses == labels[test]).mean())


# k-fold splits of `samples` for cross-validation, each a pair of train and
# held-out samples, with one part of `parts` held out in each. scoring runs
# programs on `samples` only and gets their hvs in that order, so the splits
# give places in `samples`, not sample numbers
def _folds(
    samples: np.ndarray, parts: list[np.ndarray], count: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    place = np.empty(count, dtype=int)
    place[samples] = np.arange(len(samples))
    return [
        (place[np.setdiff1d(samples, held_out)], place[held_out])
        for held_out in parts
    ]


# the held-out samples of each fold, with all the samples of a source in the
# same fold. the sources get shuffled first, so each seed gives different
# folds
def _held_out_by_source(
    samples: np.ndarray,
    sources: Sequence[Any] | np.ndarray,
    folds: int,
    count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    sources = np.asarray(sources)
    if sources.shape != (count,):
        raise ValueError(
            f"sources must give a source for each of the {count} samples in "
            "the inputs"
        )
    seen = sources[samples]
    shuffled = rng.permutation(np.unique(seen))
    if len(shuffled) < folds:
        raise ValueError(
            f"the samples come from {len(shuffled)} sources, which is fewer "
            f"than the {folds} folds"
        )
    return [
        samples[np.isin(seen, part)] for part in np.array_split(shuffled, folds)
    ]


# a few samples for `fingerprint` and `kernel`, which makes them much cheaper
# than a full run. they include the longest sample, so they see every unit
# that evolution sees
def _probe_samples(
    shuffled: np.ndarray, samples: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    probes = shuffled[:_PROBE_COUNT]
    longest = samples[np.argmax(lengths[samples])]
    if lengths[longest] > lengths[probes].max():
        probes = np.append(probes[:-1], longest)
    return probes
