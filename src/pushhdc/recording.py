"""Runs a Push program one time and replays its hv ops on many batches.

The evaluator runs each program on many batches of samples, one batch for each
length. The Push interpreter is slow for each step, and no op uses the values
or the number of units of an hv to choose what to do. So the interpreter runs
the program only one time, on stand-in hvs that record each op. Then numpy
does the recorded ops on each batch. The result is the same as a full run of
the interpreter on each batch, and ops whose result does not get to the output
are not done.
"""

from collections.abc import Callable, Sequence
from typing import Any

from pushhdc.hypervector import HyperVector
from pushhdc.inputs import Lookup
from pushhdc.pyshgp.push.instruction import Instruction, SimpleInstruction
from pushhdc.pyshgp.push.instruction_set import InstructionSet
from pushhdc.pyshgp.push.interpreter import PushInterpreter
from pushhdc.pyshgp.push.program import Program
from pushhdc.pyshgp.utils import Token


class Step:
    """one op that the program did, with the function that does it and its
    arguments. an argument is a `StandIn` hv or a value from another stack,
    for example an int. `reads` is how many times later steps and the output
    read its results."""

    def __init__(self, f: Callable[..., Any], args: tuple[Any, ...]) -> None:
        self.f = f
        self.args = args
        self.reads = 0


class StandIn(HyperVector):
    """a stand-in hv. it is result number `index` of `step`, or input number
    `index` if `step` is none. it is a `HyperVector`, so the interpreter puts
    it on the hypervector stack."""

    def __init__(self, step: Step | None, index: int) -> None:
        self.step = step
        self.index = index


class Recording:
    """the steps that lead to the hv a program leaves, in the order that the
    program did them, and the stand-in of that hv."""

    def __init__(self, steps: list[Step], output: StandIn) -> None:
        self.steps = steps
        self.output = output


def record_ops(
    program: Program, instructions: InstructionSet
) -> tuple[Recording | None, int]:
    """the ops of `program` and the program's cost. none if the program
    leaves no hv."""
    steps: list[Step] = []
    interpreter = PushInterpreter(_recording_instructions(instructions, steps))
    inputs = [StandIn(None, i) for i in range(program.signature.arity)]
    [output] = interpreter.run(program, inputs)
    if output is Token.no_stack_item:
        return None, interpreter.cost
    # counts the reads of each step that leads to the output. the arguments
    # of a step get counted at its first read only, because a replay does
    # each step one time
    todo = [output]
    while todo:
        step = todo.pop().step
        if step is not None:
            step.reads += 1
            if step.reads == 1:
                todo.extend(a for a in step.args if isinstance(a, StandIn))
    used = [step for step in steps if step.reads > 0]
    return Recording(used, output), interpreter.cost


def replay(
    recording: Recording, inputs: Sequence[HyperVector | Lookup]
) -> HyperVector:
    """the hv that the recorded program leaves for `inputs`."""
    # the results of each step are kept until the last read, so no hv stays
    # in memory longer than it must
    results: dict[int, tuple[HyperVector, ...]] = {}
    reads_left: dict[int, int] = {}

    def read(hv: StandIn) -> HyperVector:
        if hv.step is None:
            batch = inputs[hv.index]
            # the hvs of a `Lookup` get made at each read, so they stay in
            # memory only while the step that reads them needs them
            if isinstance(batch, Lookup):
                return HyperVector(batch.array())
            return batch
        key = id(hv.step)
        value = results[key][hv.index]
        reads_left[key] -= 1
        if reads_left[key] == 0:
            del results[key], reads_left[key]
        return value

    # each step comes after the steps it reads, because the program did them
    # in this order
    for step in recording.steps:
        args = [read(a) if isinstance(a, StandIn) else a for a in step.args]
        results[id(step)] = step.f(*args)
        reads_left[id(step)] = step.reads
    return read(recording.output)


# a copy of `instructions` where each op that takes an hv adds a step to
# `steps` and gives stand-ins. it keeps each op that takes no hv as it is
def _recording_instructions(
    instructions: InstructionSet, steps: list[Step]
) -> InstructionSet:
    recording = InstructionSet(type_library=instructions.type_library)
    for name, instruction in instructions.items():
        takes = getattr(instruction, "input_stacks", [])
        recording[name] = (
            _recording_op(instruction, steps)
            if "hypervector" in takes
            else instruction
        )
    return recording


# an op that takes an hv and gives a value to another stack could change what
# the program does next, so it cannot be recorded. a recording also needs
# each op that takes hvs to never choose what to do by their values. the hv
# ops never do, so only the stacks get checked here
def _recording_op(
    instruction: Instruction, steps: list[Step]
) -> SimpleInstruction:
    gives = getattr(instruction, "output_stacks", [])
    if not isinstance(instruction, SimpleInstruction) or any(
        stack != "hypervector" for stack in gives
    ):
        raise ValueError(
            f"{instruction.name} cannot be recorded, because only ops that "
            "take hvs and give only hvs can be recorded"
        )

    def record_step(*args: Any) -> tuple[StandIn, ...]:
        step = Step(instruction.f, args)
        steps.append(step)
        return tuple(StandIn(step, i) for i in range(len(gives)))

    return SimpleInstruction(
        instruction.name,
        record_step,
        instruction.input_stacks,
        gives,
        instruction.code_blocks,
        cost=instruction.cost,
    )
