"""Tests for running a Push program one time and replaying its hv ops."""

import tracemalloc
from typing import Any

import numpy as np
import pytest

from pushhdc.evaluation import Evaluator
from pushhdc.hypervector import HyperVector
from pushhdc.push_hdc import hdc_instruction_set
from pushhdc.pyshgp.gp.genome import Genome, genome_to_code
from pushhdc.pyshgp.push.atoms import Input, InstructionMeta
from pushhdc.pyshgp.push.interpreter import PushInterpreter
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.pyshgp.utils import Token
from pushhdc.recording import record, replay

# Helpers


def program(genome: Any, arity: int) -> Program:
    signature = ProgramSignature(arity=arity, output_stacks=["hypervector"])
    return Program(code=genome_to_code(genome), signature=signature)


# an evaluator that has every op, the unit ops and dup too, for tests that
# only need its ops
def evaluator(inputs: list[np.ndarray]) -> Evaluator:
    count = len(inputs[0])
    return Evaluator(
        inputs,
        np.zeros(count),
        np.arange(count),
        folds=2,
        seed=0,
        unit_ops=True,
    )


# Tests


def test_replay_gives_the_same_hv_and_cost_as_a_full_run() -> None:
    """random programs with every op of the evaluator, the unit ops and dup
    too, give the same hv and cost from a replay as from a full run.

    fails if a replay leaves out, repeats or reorders an op, gives an op the
    wrong hv or int, or if the recording loses the cost of an op.
    """
    rng = np.random.default_rng(0)
    inputs = [
        rng.choice([-1, 1], size=(4, 6, 32)),
        rng.choice([-1, 1], size=(1, 6, 32)),
    ]
    ops = evaluator(inputs)
    hvs = [HyperVector(batch) for batch in inputs]
    spawner = ops.spawner()
    for _ in range(200):
        genome = program(spawner.spawn_genome((10, 50), rng=rng), arity=2)
        interpreter = PushInterpreter(ops.instructions)
        [full] = interpreter.run(genome, hvs)
        output, cost = record(genome, ops.instructions)
        assert cost == interpreter.cost
        if full is Token.no_stack_item:
            assert output is None
        else:
            assert output is not None
            assert np.array_equal(replay(output, hvs).data, full.data)


def test_replay_does_only_the_ops_that_lead_to_the_output() -> None:
    """a replay does each op that leads to the output one time, also when a
    dup lets two ops read the same result, and no other op.

    fails if a replay does an op again for each read of its result, which in
    a chain of dups doubles the work at each dup, or if it does ops whose
    result the program does not use.
    """
    ones = np.ones((1, 1, 8))
    instructions = evaluator([ones]).instructions
    bind = instructions["hv_bind"].f
    calls = []

    def counted(a: HyperVector, b: HyperVector) -> tuple[HyperVector]:
        calls.append(1)
        result: tuple[HyperVector] = bind(a, b)
        return result

    instructions["hv_bind"].f = counted
    chain = [Input(input_index=0)]
    for _ in range(12):
        chain += [
            InstructionMeta(name="hypervector_dup", code_blocks=0),
            InstructionMeta(name="hv_bind", code_blocks=0),
        ]
    # the second program leaves its input on top of the chain's result
    for genes, binds in ((chain, 12), ([*chain, Input(input_index=0)], 0)):
        calls.clear()
        output, _ = record(program(Genome(genes), arity=1), instructions)
        assert output is not None
        replay(output, [HyperVector(ones)])
        assert len(calls) == binds


def test_replay_keeps_an_hv_only_until_its_last_read() -> None:
    """during a replay of a chain of binds, only the hvs that later ops still
    read take memory, as in a full run of the interpreter.

    fails if a replay keeps every result to the end. many worker processes
    that each replay a long program on a big batch could then use up the
    memory.
    """
    big = np.ones((1, 1, 1_000_000), dtype=np.int8)
    genes = [Input(input_index=0)]
    for _ in range(20):
        genes += [
            Input(input_index=0),
            InstructionMeta(name="hv_bind", code_blocks=0),
        ]
    instructions = evaluator([np.ones((4, 1, 8))]).instructions
    output, _ = record(program(Genome(genes), arity=1), instructions)
    assert output is not None
    tracemalloc.start()
    replay(output, [HyperVector(big)])
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # a bind can use a few hvs of memory for a short time. the chain makes
    # 20 results, so keeping all of them would use much more
    assert peak < 10 * big.nbytes


def test_an_op_that_gives_a_value_from_hvs_cannot_be_recorded() -> None:
    """recording refuses an op that turns hvs into a value on another stack.

    fails if the recording accepts such an op. its value could change what
    the program does next, so a replay could be wrong with no error.
    """
    instructions = hdc_instruction_set(register_core=False)
    with pytest.raises(ValueError, match="hv_cosine_sim"):
        record(program(Genome([]), arity=0), instructions)
