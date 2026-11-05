"""Tests for interpreter error handling and its step, cost and error counts."""

from pushhdc.pyshgp.push.atoms import CodeBlock, InstructionMeta, Literal
from pushhdc.pyshgp.push.instruction import SimpleInstruction
from pushhdc.pyshgp.push.instruction_set import InstructionSet
from pushhdc.pyshgp.push.interpreter import (
    PushInterpreter,
    PushInterpreterStatus,
)
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.pyshgp.push.types import PushInt

# Helpers


def explode(_):
    raise RuntimeError("boom")


# an interpreter with three int instructions. `paid_add` costs 1 so the cost
# count can be checked, `free_dup` costs 0, and `explode` always raises
def make_interpreter():
    instruction_set = InstructionSet(register_core=False)
    instruction_set.register_list([
        SimpleInstruction(
            "paid_add", lambda a, b: (a + b,), ["int", "int"], ["int"],
            code_blocks=0, cost=1,
        ),
        SimpleInstruction(
            "free_dup", lambda a: (a, a), ["int"], ["int", "int"],
            code_blocks=0,
        ),
        SimpleInstruction("explode", explode, ["int"], ["int"], code_blocks=0),
    ])
    return PushInterpreter(instruction_set)


def make_program(*atoms):
    return Program(
        code=CodeBlock(list(atoms)),
        signature=ProgramSignature(arity=0, output_stacks=["int"]),
    )


def lit(value):
    return Literal(value=value, push_type=PushInt)


def instr(name):
    return InstructionMeta(name=name, code_blocks=0)


# Tests


def test_instruction_that_raises_is_skipped():
    """an instruction that raises is counted and skipped, and the run goes on.

    fails if the exception stops the run, or if it isn't counted in `errors`.
    """
    interpreter = make_interpreter()
    program = make_program(lit(1), instr("explode"), lit(2))

    assert interpreter.run(program, inputs=[]) == [2]
    assert interpreter.status == PushInterpreterStatus.normal
    assert interpreter.errors == 1


def test_counts_match_a_hand_counted_program():
    """steps, cost and errors match a program counted by hand, on every run.

    the first `paid_add` has nothing to add, so it does nothing but still
    costs 1. steps include the outer code block. fails if cost only counts
    instructions that did work, ignores each instruction's `cost`, or if the
    counts carry over from one run to the next.
    """
    interpreter = make_interpreter()
    program = make_program(
        instr("paid_add"), lit(1), lit(2), instr("paid_add"), instr("free_dup")
    )

    # the second run checks the counts reset instead of carrying over
    for _ in range(2):
        assert interpreter.run(program, inputs=[]) == [3]
        counts = (interpreter.steps, interpreter.cost, interpreter.errors)
        assert counts == (6, 2, 0)
