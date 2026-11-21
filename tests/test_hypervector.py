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


def test_ops_clip_instead_of_wrapping() -> None:
    """bind and bundle clip big values to the int8 range.

    fails if they wrap around, where 16 * 16 comes out as 0 and 100 + 100 as
    -56, which would score programs on wrong values.
    """
    hv = HyperVector(np.array([16, 100, -100]))
    assert hv.bind(hv).data.tolist() == [127, 127, 127]
    assert hv.bundle(hv).data.tolist() == [32, 127, -127]


def test_ops_on_small_values_give_the_same_values_as_on_big_ones() -> None:
    """bind and bundle give the clipped value for every pair of values, also
    at the edge where int8 math starts to wrap around, and for an input that
    all samples share.

    fails if the ops use int8 math when a product or sum can pass 127, for
    example when the size of -128 is found with int8 math, where it comes out
    as -128.
    """
    edges = [-128, -127, -64, -2, -1, 0, 1, 2, 63, 64, 127]
    for x in edges:
        for y in edges:
            # shaped (samples, units, dimension), and `a` is shared
            a = np.array([x, 1, -1], dtype=np.int8).reshape(1, 1, 3)
            b = np.array([[y, -1, 1], [1, y, 0]], dtype=np.int8)[:, None]
            wide = a.astype(np.int16)
            bound = HyperVector(a).bind(HyperVector(b)).data
            summed = HyperVector(a).bundle(HyperVector(b)).data
            assert bound.tolist() == np.clip(wide * b, -127, 127).tolist()
            assert summed.tolist() == np.clip(wide + b, -127, 127).tolist()


def test_ops_leave_their_inputs_as_they_are() -> None:
    """bind, bundle and permute give a new hv and do not change their inputs,
    with small values and with values that must clip.

    fails if an op changes an input in place. an hv uses an int8 array as it
    is, with no copy, so that would also change the array it came from, for
    example an input that the evaluator gives to every program.
    """
    for size in (1, 100):
        a = np.array([size, -1, 1], dtype=np.int8)
        b = np.array([-1, -size, 1], dtype=np.int8)
        hv_a, hv_b = HyperVector(a), HyperVector(b)
        hv_a.bind(hv_b)
        hv_a.bundle(hv_b)
        hv_a.permute(1)
        assert a.tolist() == [size, -1, 1]
        assert b.tolist() == [-1, -size, 1]


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
