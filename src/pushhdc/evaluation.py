"""Scores Push programs as HDC encoders, for `evolve` to rank."""

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from pushhdc.clearing import Kernel
from pushhdc.encoder_ops import encoder_instructions, encoder_spawner
from pushhdc.fitness import Fitness
from pushhdc.inputs import Lookup, SameLength, by_length, checked_lengths
from pushhdc.pyshgp.gp.genome import GeneSpawner, genome_to_code
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.recording import record, replay

# enough kernels for a few generations of a usual population
_KEPT_KERNELS = 1024

# how many samples `fingerprint` and `kernel` run on
_PROBE_COUNT = 64


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
        # `evolve` asks for the fingerprint of each new genome, then for its
        # kernel. both come from the same probe hvs, so `fingerprint` keeps
        # the kernels of the latest genomes for `kernel` to read
        self._kernels: dict[Any, Kernel | None] = {}
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
        fingerprint. it also keeps the kernel of the probe hvs for `kernel`."""
        hvs, cost = self._run(genome, self.probes_by_length)
        self._keep_kernel(genome, hvs)
        if hvs is None:
            return b"", cost
        return hashlib.sha256(hvs.tobytes()).digest(), cost

    def kernel(self, genome: Any) -> Kernel | None:
        """how alike the program makes each pair of probe samples, for
        `evolve` to tell programs apart by. none if the program tells no
        probes apart."""
        if genome not in self._kernels:
            hvs, _ = self._run(genome, self.probes_by_length)
            self._keep_kernel(genome, hvs)
        return self._kernels[genome]

    # keeps only the latest kernels, so that the memory they use stays small
    def _keep_kernel(self, genome: Any, hvs: np.ndarray | None) -> None:
        if genome not in self._kernels and len(self._kernels) >= _KEPT_KERNELS:
            del self._kernels[next(iter(self._kernels))]
        self._kernels[genome] = None if hvs is None else Kernel.of(hvs)

    # the kept kernels only help the process that made them. so a copy of the
    # evaluator for a worker process leaves them out
    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_kernels": {}}

    # scoring, `encode`, `fingerprint` and `kernel` run programs the same way,
    # just on different samples
    def _run(
        self, genome: Any, parts: list[SameLength]
    ) -> tuple[np.ndarray | None, int]:
        signature = ProgramSignature(
            arity=self.arity, output_stacks=["hypervector"]
        )
        program = Program(code=genome_to_code(genome), signature=signature)
        # every length does the same ops, so the interpreter runs one time and
        # numpy does the ops again on each length (see `pushhdc.recording`)
        output, cost = record(program, self.instructions)
        if output is None:
            return None, cost
        sums = []
        for where, inputs in parts:
            hvs = replay(output, inputs).data.sum(axis=1)
            # a program that only uses inputs shared by all samples gives one
            # hv, which is then the hv of every sample
            if len(hvs) < len(where):
                hvs = np.repeat(hvs, len(where), axis=0)
            sums.append(hvs)
        # puts each hv back at the place of its sample
        order = np.concatenate([where for where, _ in parts])
        return np.concatenate(sums)[np.argsort(order)], cost

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
