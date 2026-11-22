"""Scores Push programs as HDC encoders, for `evolve` to rank."""

import hashlib
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from pushhdc.clearing import Kernel
from pushhdc.encoder_ops import encoder_instructions, encoder_spawner
from pushhdc.fitness import Fitness
from pushhdc.inputs import Lookup, SameLength, by_length, checked_lengths
from pushhdc.pyshgp.gp.genome import GeneSpawner, genome_to_code
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.recording import record_ops, replay

# the evaluator keeps the probe results of this many genomes, enough for a
# few generations of a usual population
_KEPT_PROBE_RESULTS = 1024

# how many samples `fingerprint` and `kernel` run on
_PROBE_COUNT = 64


# what `Evaluator` keeps for each genome that it ran on the probe samples
class _ProbeResult(NamedTuple):
    fingerprint: bytes
    cost: int
    kernel: Kernel | None


class Evaluator:
    """scores a genome as an encoder, by cross-validated accuracy and cost.

    each input is a batch of hypervectors shaped (samples, units, dimension),
    where a unit is one piece of a sample. a program does the same ops on each
    unit, so a short program can encode every piece the same way. then the
    units of each sample get bundled into one hv. with `unit_ops`, a program
    can also combine each unit with a later unit, and keep only some units.
    evolution then chooses how the units make groups. the units must be in
    order, for example sequence positions.

    an input that is the same for every sample, for example the hv of each
    position, can have 1 in place of the number of samples. it then uses less
    memory and makes programs faster, and the scores do not change.

    an input can also be a `Lookup`, one hv for each symbol and the symbol at
    each unit. then the padded array never gets made, and the scores are the
    same as with it.

    `lengths` gives the number of units of each sample in `inputs`, the test
    samples too. sample i then has only its first `lengths[i]` units in each
    input, the shared inputs too. the units after that are left out, so they
    can hold any values, and a sample gets the same hv that it gets alone.
    the program runs one time for each different length, so many different
    lengths make scoring slower.

    `samples` are the ones evolution may see, so everything but the test
    split. k-fold cross-validation scores each of the `folds` parts with
    prototypes built from the others, so every sample counts toward accuracy
    and none is judged by a prototype it helped build.
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
    ) -> None:
        self.arity = len(inputs)
        lengths = checked_lengths(inputs, lengths)
        # the ints of the unit operations reach the last unit of the longest
        # sample that evolution sees
        self.units = int(lengths[samples].max())
        # shuffled in case samples are sorted by class, where one fold could
        # hold most of a class. done once so every genome gets the same folds
        shuffled = np.random.default_rng(seed).permutation(samples)
        self.splits = _folds(samples, shuffled, folds, len(lengths))
        self.seen_labels = labels[samples]
        # the other samples, for example test, only get encoded at the end. so
        # they are kept apart, and scoring does not run programs on them
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
            units=self.units,
        )

    def encode(self, genome: Any) -> tuple[np.ndarray | None, int]:
        """each sample's hv, shaped (samples, dimension), and the program's
        cost, for scoring other splits like test at the end. no hvs if the
        program leaves nothing on the hypervector stack."""
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

    def fingerprint(self, genome: Any) -> tuple[bytes, int]:
        """a fingerprint that programs share when they encode a few probe
        samples the same way, and the program's cost. programs with the same
        fingerprint get the same accuracy, so `evolve` only scores one of
        them. programs that leave no hv all score 0, so they share an empty
        fingerprint."""
        result = self._probe(genome)
        return result.fingerprint, result.cost

    def kernel(self, genome: Any) -> Kernel | None:
        """how alike the program makes each pair of probe samples, for
        `evolve` to tell programs apart by. none if the program tells no
        probes apart."""
        return self._probe(genome).kernel

    # `evolve` asks for the fingerprint and then the kernel of each new genome.
    # both come from one run on the probe samples, so `_probe` keeps the
    # results of the latest genomes, and drops the oldest so that the memory
    # stays small
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

    # the kept probe results only help the process that made them. so a copy
    # of the evaluator for a worker process leaves them out
    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_probe_results": {}}

    # scoring, `encode`, `fingerprint` and `kernel` run programs the same way,
    # just on different samples. the hvs come back in the order of the rows
    # that the groups were cut from
    def _run(
        self, genome: Any, groups: list[SameLength]
    ) -> tuple[np.ndarray | None, int]:
        signature = ProgramSignature(
            arity=self.arity, output_stacks=["hypervector"]
        )
        program = Program(code=genome_to_code(genome), signature=signature)
        # every length does the same ops, so the interpreter runs one time and
        # numpy does the ops again on each length (see `pushhdc.recording`)
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
    # a class whose train hvs are all zero has no direction. its prototype
    # stays zero, so every sample is 0 similar to it
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes /= np.where(norms == 0, 1, norms)
    guesses = classes[(hvs[test] @ prototypes.T).argmax(axis=1)]
    return float((guesses == labels[test]).mean())


# k-fold splits of `samples` for cross-validation, each a pair of train and
# held-out samples. scoring runs programs on `samples` only and gets their hvs
# in that order, so the splits give places in `samples`, not sample numbers
def _folds(
    samples: np.ndarray, shuffled: np.ndarray, folds: int, count: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    place = np.empty(count, dtype=int)
    place[samples] = np.arange(len(samples))
    return [
        (place[np.setdiff1d(samples, held_out)], place[held_out])
        for held_out in np.array_split(shuffled, folds)
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
