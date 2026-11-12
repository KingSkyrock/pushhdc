"""Smoke helpers for Push + HDC wiring."""

from __future__ import annotations

import logging

import numpy as np

from pushhdc.hypervector import HyperVector
from pushhdc.push_hdc import hdc_instruction_set
from pushhdc.pyshgp.push.atoms import CodeBlock, Literal
from pushhdc.pyshgp.push.config import PushConfig
from pushhdc.pyshgp.push.interpreter import PushInterpreter
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.pyshgp.push.type_library import PushTypeLibrary

logger = logging.getLogger(__name__)


def run_pyshgp_smoke_test() -> list[int]:
    """Run a minimal Push program and return its integer output."""
    type_library = PushTypeLibrary()
    program = Program(
        code=CodeBlock([Literal(value=42, push_type=type_library["int"])]),
        signature=ProgramSignature(arity=0, output_stacks=["int"]),
    )
    output = PushInterpreter().run(program, inputs=[])
    if output != [42]:
        raise RuntimeError(f"Unexpected PyshGP output: {output!r}")
    return [int(value) for value in output]


def run_hdc_smoke_test(dimension: int = 256) -> float:
    """Bind two HVs on the Push stack and return cosine(sim of result, first).

    Exercises hypervector type registration, stack I/O, and ``hv_bind``.
    """
    rng = np.random.default_rng(0)
    instruction_set = hdc_instruction_set()
    type_library = instruction_set.type_library
    hv_type = type_library["hypervector"]

    hv_a = HyperVector.random(dimension=dimension, rng=rng)
    hv_b = HyperVector.random(dimension=dimension, rng=rng)

    program = Program(
        code=CodeBlock(
            [
                Literal(value=hv_a, push_type=hv_type),
                Literal(value=hv_b, push_type=hv_type),
                instruction_set["hv_bind"].meta(),
                Literal(value=hv_a, push_type=hv_type),
                instruction_set["hv_cosine_sim"].meta(),
            ]
        ),
        signature=ProgramSignature(
            arity=0,
            output_stacks=["float"],
            push_config=PushConfig(rng=rng),
        ),
    )
    output = PushInterpreter(instruction_set).run(program, inputs=[])
    if len(output) != 1 or not isinstance(output[0], float):
        raise RuntimeError(f"Unexpected HDC Push output: {output!r}")
    return float(output[0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pyshgp_smoke_test()
    sim = run_hdc_smoke_test()
    logger.info("hdc smoke cosine(bound, a)=%.4f", sim)
