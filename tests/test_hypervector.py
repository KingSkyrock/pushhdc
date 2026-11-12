"""Tests for hypervector ops, on their own and inside Push programs."""

import numpy as np
import pytest

from pushhdc.hypervector import HyperVector
from pushhdc.push_hdc import hdc_instruction_set
from pushhdc.pyshgp.push.atoms import CodeBlock, Input
from pushhdc.pyshgp.push.interpreter import PushInterpreter
from pushhdc.pyshgp.push.program import Program, ProgramSignature

# Tests


def test_random_hypervector_is_reproducible_with_seed() -> None:
    """the same seed gives the same random hypervector.

    fails if `random` draws from its own unseeded generator instead of
    `rng`.
    """
    first = HyperVector.random(1000, rng=np.random.default_rng(0))
    again = HyperVector.random(1000, rng=np.random.default_rng(0))
    assert np.array_equal(first.data, again.data)


def test_push_program_measures_similarity_of_bound_hv() -> None:
    """a Push program can bind hvs and push their cosine similarity.

    binding `a` with `b` flips `a` wherever `b` is -1, so the result's cosine
    with `a` is the mean of `b`. fails if `hv_cosine_sim` isn't normalized or
    leaves nothing on the float stack, or if `hv_bind` doesn't multiply.
    """
    rng = np.random.default_rng(0)
    a = HyperVector.random(256, rng=rng)
    b = HyperVector.random(256, rng=rng)
    instructions = hdc_instruction_set()
    code = CodeBlock(
        [
            Input(input_index=0),
            Input(input_index=1),
            instructions["hv_bind"].meta(),
            Input(input_index=0),
            instructions["hv_cosine_sim"].meta(),
        ]
    )
    signature = ProgramSignature(arity=2, output_stacks=["float"])
    output = PushInterpreter(instructions).run(
        Program(code=code, signature=signature), [a, b]
    )
    assert output == pytest.approx([b.data.mean()])
