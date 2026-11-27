"""Helpers that more than one test file uses."""

from typing import Any

import numpy as np

from pushhdc.evaluation import _PROBE_COUNT, Evaluator
from pushhdc.inputs import Lookup
from pushhdc.pyshgp.gp.genome import Genome
from pushhdc.pyshgp.push.atoms import Input, InstructionMeta, Literal
from pushhdc.pyshgp.push.types import PushInt

# as many samples as `Evaluator` runs a program on for its fingerprints and
# kernels. it comes from `Evaluator`, so that the tests follow a change to it
PROBE_COUNT = _PROBE_COUNT


# lets tests write genomes as names like "input_0" and "hv_bind". ints become
# int literals
def genome(*genes: str | int) -> Any:
    def gene(name: str | int) -> Any:
        if isinstance(name, int):
            return Literal(value=name, push_type=PushInt)
        if name.startswith("input_"):
            return Input(input_index=int(name.removeprefix("input_")))
        return InstructionMeta(name=name, code_blocks=0)

    return Genome([gene(name) for name in genes])


# random hvs of -1 and 1, one for each of as many samples as the probes
def random_hvs(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice([-1, 1], size=(PROBE_COUNT, 256))


# the same hvs after a fixed bind and permute, plus a bias for every sample.
# a program with these does the same thing as one with `hvs`
def same_thing(hvs: np.ndarray) -> np.ndarray:
    key = np.random.default_rng(100).choice([-1, 1], size=hvs.shape[1])
    return np.roll(hvs * key, 3, axis=1) + 5


# an evaluator for tests of one program, where the labels and folds do not
# matter
def evaluator_on(
    *inputs: np.ndarray | Lookup,
    unit_ops: bool = False,
    lengths: np.ndarray | None = None,
) -> Evaluator:
    samples = np.arange(len(inputs[0]))
    labels = np.zeros(len(samples))
    return Evaluator(
        inputs,
        labels,
        samples,
        folds=2,
        seed=0,
        unit_ops=unit_ops,
        lengths=lengths,
    )


# samples with random values in every unit, so the units past each length
# hold values that must not count. the lengths are out of order, and one
# length comes twice with another length between, so samples of one length
# are not next to each other. the lengths include 0 and all the units. one
# input is shared by all samples
def samples_of_many_lengths() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    a = rng.choice([-1, 1], size=(6, 8, 16))
    b = rng.choice([-1, 1], size=(1, 8, 16))
    return a, b, np.array([3, 0, 7, 3, 8, 1])


# the first program shifts by one, so the last unit of each sample reads the
# unit just past its end, which must count as zero. the second uses only the
# shared input, so it gives one hv for each length, and every sample of that
# length must get it
PROGRAMS_FOR_LENGTHS = (
    genome("input_0", 1, "hv_shift_units", "input_1", "hv_bind"),
    genome("input_1"),
)
