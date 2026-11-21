"""Scores Push programs as HDC encoders, for `evolve` to rank."""

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from pushhdc.clearing import Kernel
from pushhdc.evolution import Fitness
from pushhdc.hypervector import HyperVector
from pushhdc.push_hdc import hdc_instruction_set
from pushhdc.pyshgp.gp.genome import GeneSpawner, genome_to_code
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.recording import record, replay
from pushhdc.units import unit_instructions

# enough kernels for a few generations of a usual population
_KEPT_KERNELS = 1024

# samples of one length, which a program runs on together. the array says
# where each of them goes in the output, and the hvs are their inputs, cut to
# that length
type SameLength = tuple[np.ndarray, list[HyperVector]]


class Lookup:
    """an input given as one hv for each symbol and the symbol at each unit,
    so nobody has to build the padded (samples, units, dimension) array.

    unit u of sample i is `hvs[symbols[i, u]]`. `hvs` has one row for each
    symbol, and `symbols` is shaped (samples, units), or (1, units) for an
    input that every sample shares. the evaluator makes each unit's hv when it
    cuts its inputs, and with `lengths` it makes only the units within each
    length. for strings, the padded array holds a copy of a letter's hv at
    each place up to the longest string, so it can be much bigger than the
    hvs and symbols. the symbols past each length are never read, but each
    must still pick an hv, for example 0.
    """

    ndim = 3

    def __init__(self, hvs: np.ndarray, symbols: np.ndarray) -> None:
        hvs, symbols = np.asarray(hvs), np.asarray(symbols)
        if hvs.ndim != 2 or symbols.ndim != 2:
            raise ValueError(
                "hvs must have one row for each symbol, and symbols must be "
                "shaped (samples, units)"
            )
        # numpy would take -1 as the last hv with no error
        if not np.issubdtype(symbols.dtype, np.integer) or (
            symbols.size > 0
            and (symbols.min() < 0 or symbols.max() >= len(hvs))
        ):
            raise ValueError(
                f"symbols must pick one of the {len(hvs)} hvs, so they must be "
                f"whole numbers from 0 to {len(hvs) - 1}"
            )
        self.hvs = hvs
        self.symbols = symbols
        self.shape = (*symbols.shape, hvs.shape[1])

    def __len__(self) -> int:
        return len(self.symbols)

    # the evaluator reads an input in cuts of samples and units, so only the
    # hvs of those units get made
    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self.hvs[self.symbols[key]])


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
    samples too, for example the letters of each string. sample i then has
    only its first `lengths[i]` units in each input, the shared inputs too.
    the units after that are left out, so they can hold any values, and a
    sample gets the same hv that it gets alone. the program runs one time for
    each different length, so many different lengths make scoring slower.

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
        if any(batch.ndim != 3 for batch in inputs):
            raise ValueError(
                "inputs must be shaped (samples, units, dimension)"
            )
        self.arity = len(inputs)
        count = max(len(batch) for batch in inputs)
        units = inputs[0].shape[1]
        lengths = (
            np.full(count, units) if lengths is None else np.asarray(lengths)
        )
        if (
            not np.issubdtype(lengths.dtype, np.integer)
            or lengths.shape != (count,)
            or lengths.min() < 0
            or lengths.max() > units
        ):
            raise ValueError(
                f"lengths must give each of the {count} samples in the inputs "
                f"a whole number of units from 0 to {units}"
            )
        # the ints of the unit operations reach the last unit of the longest
        # sample that evolution sees
        self.units = int(lengths[samples].max())
        # shuffled in case samples are sorted by class, where one fold could
        # hold most of a class. done once so every genome gets the same folds
        shuffled = np.random.default_rng(seed).permutation(samples)
        # scoring runs programs on `samples` only and gets their hvs in that
        # order. so the folds and labels go by place in `samples`, not by
        # sample number
        place = np.empty(count, dtype=int)
        place[samples] = np.arange(len(samples))
        self.splits = [
            (place[np.setdiff1d(samples, held_out)], place[held_out])
            for held_out in np.array_split(shuffled, folds)
        ]
        self.labels = labels[samples]
        # a few samples for `fingerprint` and `kernel`, which makes them much
        # cheaper than a full run. they include the longest sample, so they
        # see every unit that evolution sees
        probes = shuffled[:64]
        longest = samples[np.argmax(lengths[samples])]
        if lengths[longest] > lengths[probes].max():
            probes = np.append(probes[:-1], longest)
        # the other samples, for example test, only get encoded at the end. so
        # they are kept apart, and scoring does not run programs on them
        self.samples = samples
        self.others = np.setdiff1d(np.arange(count), samples)
        self.seen = _by_length(inputs, lengths, samples)
        self.unseen = _by_length(inputs, lengths, self.others)
        self.probes = _by_length(inputs, lengths, probes)
        # `evolve` asks for the fingerprint of each new genome, then for its
        # kernel. both come from the same probe hvs, so `fingerprint` keeps
        # the kernels of the latest genomes for `kernel` to read
        self._kernels: dict[Any, Kernel | None] = {}
        # an encoder only needs the hv ops, since similarity is the
        # classifier's job
        self.instructions = hdc_instruction_set(register_core=False)
        self.instructions.unregister("hv_cosine_sim")
        self.unit_ops = unit_ops
        if unit_ops:
            self.instructions.register_list(unit_instructions())
            # a program can use one result more than one time, and does not
            # have to make it again
            self.instructions.register_core_by_name("hypervector_dup$")
        # every op that does arithmetic has the same cost
        for instruction in self.instructions.values():
            instruction.cost = 1
        # ops that only move data or leave units out have no cost, because
        # hardware does them with wiring and no arithmetic
        self.instructions["hv_permute"].cost = 0
        if unit_ops:
            self.instructions["hv_shift_units"].cost = 0
            self.instructions["hv_keep_units"].cost = 0
            # a copy of an hv has no cost, because hardware can use the same hv
            # again
            self.instructions["hypervector_dup"].cost = 0

    def spawner(self) -> GeneSpawner:
        """makes random genes that match this evaluator's inputs and ops."""
        if self.unit_ops:
            # the unit operations use ints for the shift distance and for the
            # two ends of a range. `hv_permute` uses the same ints. the ints go
            # from 0 to the last unit, so a range can start at the first unit
            units = self.units
            return GeneSpawner(
                n_inputs=self.arity,
                instruction_set=self.instructions,
                literals=[],
                erc_generators=[lambda rng: int(rng.integers(0, units))],
            )
        # hv_permute takes its shift from the int stack. permuting twice
        # shifts by two, so 1 can reach any shift
        return GeneSpawner(
            n_inputs=self.arity,
            instruction_set=self.instructions,
            literals=[1],
            erc_generators=[],
        )

    def encode(self, genome: Any) -> tuple[np.ndarray | None, int]:
        """each sample's hv, shaped (samples, dimension), and the program's
        cost, for scoring other splits like test at the end. no hvs if the
        program leaves nothing on the hypervector stack."""
        seen, cost = self._run(genome, self.seen)
        if seen is None:
            return None, cost
        count = len(self.samples) + len(self.others)
        hvs = np.empty((count, *seen.shape[1:]), seen.dtype)
        hvs[self.samples] = seen
        if len(self.others):
            hvs[self.others], _ = self._run(genome, self.unseen)
        return hvs, cost

    def fingerprint(self, genome: Any) -> tuple[bytes, int]:
        """a fingerprint that programs share when they encode a few probe
        samples the same way, and the program's cost. programs with the same
        fingerprint get the same accuracy, so `evolve` only scores one of
        them. programs that leave no hv all score 0, so they share an empty
        fingerprint."""
        hvs, cost = self._run(genome, self.probes)
        self._keep_kernel(genome, hvs)
        if hvs is None:
            return b"", cost
        return hashlib.sha256(hvs.tobytes()).digest(), cost

    def kernel(self, genome: Any) -> Kernel | None:
        """how alike the program makes each pair of probe samples, for
        `evolve` to tell programs apart by. none if the program tells no
        probes apart."""
        if genome not in self._kernels:
            hvs, _ = self._run(genome, self.probes)
            self._keep_kernel(genome, hvs)
        return self._kernels[genome]

    # keeps only the latest kernels, so that the memory they use stays small
    def _keep_kernel(self, genome: Any, hvs: np.ndarray | None) -> None:
        if len(self._kernels) >= _KEPT_KERNELS:
            del self._kernels[next(iter(self._kernels))]
        self._kernels[genome] = None if hvs is None else Kernel.of(hvs)

    # the kept kernels only help the process that made them. so a copy of the
    # evaluator for a worker process leaves them out
    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_kernels": {}}

    # scoring, `encode`, `fingerprint` and `kernel` run programs the same way,
    # just on different samples
    def _run(
        self, genome: Any, by_length: list[SameLength]
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
        for where, inputs in by_length:
            hvs = replay(output, inputs).data.sum(axis=1)
            # a program that only uses inputs shared by all samples gives one
            # hv, which is then the hv of every sample
            if len(hvs) < len(where):
                hvs = np.repeat(hvs, len(where), axis=0)
            sums.append(hvs)
        # puts each hv back at the place of its sample
        order = np.concatenate([where for where, _ in by_length])
        return np.concatenate(sums)[np.argsort(order)], cost

    def __call__(self, genome: Any) -> Fitness:
        hvs, cost = self._run(genome, self.seen)
        # hvs that are all zero contain no information
        if hvs is None or not hvs.any():
            return Fitness(accuracy=0.0, cost=cost)
        accuracies = [
            centroid_accuracy(hvs, self.labels, train, held_out)
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


# puts the samples in `rows` that have the same length together, and cuts
# each of their inputs to that length. a shared input stays shared. the array
# of each length says where its samples are in `rows`
def _by_length(
    inputs: Sequence[np.ndarray | Lookup], lengths: np.ndarray, rows: np.ndarray
) -> list[SameLength]:
    out = []
    for length in np.unique(lengths[rows]):
        where = np.flatnonzero(lengths[rows] == length)
        cut = [
            batch[:, :length]
            if len(batch) == 1
            else batch[rows[where], :length]
            for batch in inputs
        ]
        out.append((where, [HyperVector(batch) for batch in cut]))
    return out
