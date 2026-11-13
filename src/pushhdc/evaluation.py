"""Scores Push programs as HDC encoders, for `evolve` to rank."""

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from pushhdc.evolution import Fitness
from pushhdc.hypervector import HyperVector
from pushhdc.push_hdc import hdc_instruction_set
from pushhdc.pyshgp.gp.genome import GeneSpawner, genome_to_code
from pushhdc.pyshgp.push.interpreter import PushInterpreter
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.pyshgp.utils import Token
from pushhdc.units import unit_instructions


class Evaluator:
    """scores a genome as an encoder, by cross-validated accuracy and cost.

    each input is a batch of hypervectors shaped (samples, units, dimension),
    where a unit is one piece of a sample. a program does the same ops on each
    unit, so a short program can encode every piece the same way. then the
    units of each sample get bundled into one hv. with `unit_ops`, a program
    can also combine each unit with a later unit, and keep only some units.
    evolution then chooses how the units make groups. the units must be in
    order, for example sequence positions.

    `samples` are the ones evolution may see, so everything but the test
    split. k-fold cross-validation scores each of the `folds` parts with
    prototypes built from the others, so every sample counts toward accuracy
    and none is judged by a prototype it helped build.
    """

    def __init__(
        self,
        inputs: Sequence[np.ndarray],
        labels: np.ndarray,
        samples: np.ndarray,
        *,
        folds: int = 5,
        seed: int,
        unit_ops: bool = False,
    ) -> None:
        if any(batch.ndim != 3 for batch in inputs):
            raise ValueError(
                "inputs must be shaped (samples, units, dimension)"
            )
        self.inputs = [HyperVector(batch) for batch in inputs]
        self.labels = labels
        # shuffled in case samples are sorted by class, where one fold could
        # hold most of a class. done once so every genome gets the same folds
        shuffled = np.random.default_rng(seed).permutation(samples)
        self.splits = [
            (np.setdiff1d(samples, held_out), held_out)
            for held_out in np.array_split(shuffled, folds)
        ]
        # a few samples for `fingerprint`. in the one run we measured, 64 told
        # programs apart as well as every sample did
        self.probes = [HyperVector(batch[shuffled[:64]]) for batch in inputs]
        # an encoder only needs the hv ops, since similarity is the
        # classifier's job. each op costs 1 until we measure real costs
        self.instructions = hdc_instruction_set(register_core=False)
        self.instructions.unregister("hv_cosine_sim")
        self.unit_ops = unit_ops
        if unit_ops:
            self.instructions.register_list(unit_instructions())
            # a program can use one result more than one time, and does not
            # have to make it again
            self.instructions.register_core_by_name("hypervector_dup$")
        for instruction in self.instructions.values():
            instruction.cost = 1
        if unit_ops:
            # a copy of an hv has no cost, because hardware can use the same hv
            # again
            self.instructions["hypervector_dup"].cost = 0

    def spawner(self) -> GeneSpawner:
        """makes random genes that match this evaluator's inputs and ops."""
        if self.unit_ops:
            # the unit operations use ints for the shift distance and for the
            # two ends of a range. `hv_permute` uses the same ints. the ints go
            # from 0 to the last unit, so a range can start at the first unit
            units = self.inputs[0].data.shape[1]
            return GeneSpawner(
                n_inputs=len(self.inputs),
                instruction_set=self.instructions,
                literals=[],
                erc_generators=[lambda rng: int(rng.integers(0, units))],
            )
        # hv_permute takes its shift from the int stack. permuting twice
        # shifts by two, so 1 can reach any shift
        return GeneSpawner(
            n_inputs=len(self.inputs),
            instruction_set=self.instructions,
            literals=[1],
            erc_generators=[],
        )

    def encode(self, genome: Any) -> tuple[np.ndarray | None, int]:
        """each sample's hv, shaped (samples, dimension), and the program's
        cost, for scoring other splits like test at the end. no hvs if the
        program leaves nothing on the hypervector stack."""
        return self._run(genome, self.inputs)

    def fingerprint(self, genome: Any) -> tuple[bytes, int]:
        """a key that programs share when they encode a few probe samples the
        same way, and the program's cost. programs with the same key get the
        same accuracy, so `evolve` only scores one of them. ones that leave no
        hv all score 0, so they share an empty key."""
        hvs, cost = self._run(genome, self.probes)
        if hvs is None:
            return b"", cost
        return hashlib.sha256(hvs.tobytes()).digest(), cost

    # `encode` and `fingerprint` run programs the same way, just on different
    # samples
    def _run(
        self, genome: Any, inputs: list[HyperVector]
    ) -> tuple[np.ndarray | None, int]:
        signature = ProgramSignature(
            arity=len(inputs), output_stacks=["hypervector"]
        )
        program = Program(code=genome_to_code(genome), signature=signature)
        interpreter = PushInterpreter(self.instructions)
        [output] = interpreter.run(program, inputs)
        if output is Token.no_stack_item:
            return None, interpreter.cost
        return output.data.sum(axis=1), interpreter.cost

    def __call__(self, genome: Any) -> Fitness:
        hvs, cost = self.encode(genome)
        # hvs that are all zero contain no information. they also cause a
        # division by zero in the prototypes
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
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    guesses = classes[(hvs[test] @ prototypes.T).argmax(axis=1)]
    return float((guesses == labels[test]).mean())
