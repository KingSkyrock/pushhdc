"""Scores Push programs as HDC encoders, for `evolve` to rank."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from pushhdc.evolution import Fitness
from pushhdc.hypervector import HyperVector
from pushhdc.push_hdc import hdc_instruction_set
from pushhdc.pyshgp.gp.genome import GeneSpawner, genome_to_code
from pushhdc.pyshgp.push.config import PushConfig
from pushhdc.pyshgp.push.interpreter import PushInterpreter
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.pyshgp.utils import Token


class Evaluator:
    """scores a genome as an encoder, by cross-validated accuracy and cost.

    each input is a batch of hypervectors shaped (samples, units, dimension),
    where a unit is one piece of a sample. programs see one unit at a time, so
    a short program can encode every piece the same way, and each sample's
    units get bundled into one hv.

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
    ) -> None:
        if any(batch.ndim != 3 for batch in inputs):
            raise ValueError(
                "inputs must be shaped (samples, units, dimension)"
            )
        self.inputs = [HyperVector(batch) for batch in inputs]
        self.labels = labels
        self.seed = seed
        # shuffled in case samples are sorted by class, where one fold could
        # hold most of a class. done once so every genome gets the same folds
        shuffled = np.random.default_rng(seed).permutation(samples)
        self.splits = [
            (np.setdiff1d(samples, held_out), held_out)
            for held_out in np.array_split(shuffled, folds)
        ]
        # an encoder only needs the hv ops, since similarity is the
        # classifier's job. each op costs 1 until we measure real costs
        self.instructions = hdc_instruction_set(register_core=False)
        self.instructions.unregister("hv_cosine_sim")
        for instruction in self.instructions.values():
            instruction.cost = 1

    def spawner(self) -> GeneSpawner:
        """makes random genes that match this evaluator's inputs and ops."""
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
        # bundle breaks ties at random, so each run gets a fresh seeded rng.
        # that gives a genome the same score every time and in any worker
        signature = ProgramSignature(
            arity=len(self.inputs),
            output_stacks=["hypervector"],
            push_config=PushConfig(rng=np.random.default_rng(self.seed)),
        )
        program = Program(code=genome_to_code(genome), signature=signature)
        interpreter = PushInterpreter(self.instructions)
        [output] = interpreter.run(program, self.inputs)
        if output is Token.no_stack_item:
            return None, interpreter.cost
        return output.data.sum(axis=1), interpreter.cost

    def __call__(self, genome: Any) -> Fitness:
        hvs, cost = self.encode(genome)
        if hvs is None:
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
