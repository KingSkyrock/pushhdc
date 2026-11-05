"""The :mod:`interpreter` module defines the ``PushInterpreter`` used to run Push programs."""
import logging
import traceback
from typing import Union
import time
from enum import Enum

from pushhdc.pyshgp.push.instruction import Instruction
from pushhdc.pyshgp.push.program import Program
from pushhdc.pyshgp.push.state import PushState
from pushhdc.pyshgp.push.instruction_set import InstructionSet
from pushhdc.pyshgp.push.atoms import Atom, Closer, Literal, InstructionMeta, CodeBlock, Input
from pushhdc.pyshgp.push.config import PushConfig
from pushhdc.pyshgp.tap import tap
from pushhdc.pyshgp.validation import PushError

logger = logging.getLogger(__name__)


class PushInterpreterStatus(Enum):
    """Enum class of all potential statuses of a PushInterpreter."""

    normal = 1
    step_limit_exceeded = 2
    runtime_limit_exceeded = 3
    growth_cap_exceeded = 4


class PushInterpreter:
    """An interpreter capable of running Push programs.

    Parameters
    ----------
    instruction_set : Union[InstructionSet, str], optional
        The ``InstructionSet`` to use for executing programs. Default is "core"
        which instantiates an ``InstructionSet`` using all the core instructions.

    Attributes
    ----------
    instruction_set : InstructionSet
        The ``InstructionSet`` to use for executing programs.
    state : PushState
        The current ``PushState``. Contains one stack for each ``PushType``
        mentioned by the instructions in the instruction set.
    status : PushInterpreterStatus
        A string denoting if the interpreter has encountered a situation
        where non-standard termination was required.
    steps : int
        Number of atoms (literals, inputs, code) evaluated in the last run.
    cost : int
        Sum of the ``cost`` of every instruction attempted in the last run,
        including ones that did nothing. Used as the program's inference cost.
    errors : int
        Number of instructions that raised an exception in the last run. Each
        one is skipped as a no-op instead of stopping the run.

    """

    def __init__(self,
                 instruction_set: Union[InstructionSet, str] = "core",
                 reset_on_run: bool = True):
        self.reset_on_run = reset_on_run
        # If no instruction set given, create one and register all instructions.
        if instruction_set == "core":
            self.instruction_set = InstructionSet(register_core=True)
        else:
            self.instruction_set = instruction_set

        self.type_library = self.instruction_set.type_library

        # Initialize the PushState and status
        self.state: PushState = None
        self.status: PushInterpreterStatus = None
        self.steps = 0
        self.cost = 0
        self.errors = 0
        self._validate()

    def _validate(self):
        library_type_names = set(self.type_library.keys())
        required_stacks = self.instruction_set.required_stacks() - {"stdout", "exec", "untyped"}
        if not required_stacks <= library_type_names:
            raise ValueError(
                "PushInterpreter instruction_set and type_library are incompatible. {iset} vs {tlib}. Diff: {d}".format(
                    iset=required_stacks,
                    tlib=library_type_names,
                    d=required_stacks - library_type_names,
                ))

    def _evaluate_instruction(self, instruction: Instruction, config: PushConfig):
        self.state = instruction.evaluate(self.state, config)

    def untyped_to_typed(self):
        """Infer ``PushType`` of items on state's untyped queue and push to corresponding stacks."""
        while len(self.state.untyped) > 0:
            el = self.state.untyped.popleft()
            push_type = self.type_library.push_type_of(el, error_on_not_found=True)
            self.state[push_type.name].push(el)

    @tap
    def evaluate_atom(self, atom: Atom, config: PushConfig):
        """Evaluate an ``Atom``.

        Parameters
        ----------
        atom : Atom
            The Atom (``Literal``, ``InstructionMeta``, ``Input``, or ``CodeBlock``) to
            evaluate against the current ``PushState``.
        config : PushConfig
            The configuration of the Push program being run.

        """
        try:
            if isinstance(atom, InstructionMeta):
                instruction = self.instruction_set[atom.name]
                self.cost += instruction.cost
                try:
                    self._evaluate_instruction(instruction, config)
                except Exception:
                    self.errors += 1
                    logger.debug("%s raised, skipping it", atom.name, exc_info=True)
            elif isinstance(atom, Input):
                input_value = self.state.inputs[atom.input_index]
                self.state.untyped.append(input_value)
            elif isinstance(atom, CodeBlock):
                for a in atom[::-1]:
                    self.state["exec"].push(a)
            elif isinstance(atom, Literal):
                self.state[atom.push_type.name].push(atom.value)
            elif isinstance(atom, Closer):
                raise PushError("Closers should not be in push programs. Only genomes.")
            else:
                raise PushError("Cannot evaluate {t}, require a subclass of Atom".format(t=type(atom)))
            self.untyped_to_typed()
        except Exception as e:
            err_type = type(e)
            err_msg = str(e)
            traceback.print_exc()
            raise PushError(
                "{t} raised while evaluating {atom}. Original message: \"{m}\"".format(
                    t=err_type.__name__,
                    atom=atom,
                    m=err_msg
                ))

    @tap
    def run(self,
            program: Program,
            inputs: list,
            print_trace: bool = False) -> list:
        """Run a Push ``Program`` given some inputs and desired output ``PushTypes``.

        The general flow of this method is:
            1. Create a new push state
            2. Load the program and inputs.
            3. If the exec stack is empty, return the outputs.
            4. Else, pop the exec stack and process the atom.
            5. Return to step 3.

        Parameters
        ----------
        program : Program
            Program to run.
        inputs : list
            A sequence of values to use as inputs to the push program.
        print_trace : bool
            If True, each step of program execution will be summarized in stdout.

        Returns
        -------
        Sequence
            A sequence of values pulled from the final push state. May contain
            pyshgp.utils.Token.no_stack_item if output stacks are empty.

        """
        push_config = program.signature.push_config

        if self.reset_on_run or self.state is None:
            self.state = PushState(self.type_library, push_config)
            self.status = PushInterpreterStatus.normal

        # Setup
        self.state.load_code(program.code)
        self.state.load_inputs(inputs)
        stop_time = time.time() + push_config.runtime_limit
        self.steps = 0
        self.cost = 0
        self.errors = 0

        if print_trace:
            print("Initial State:")
            self.state.pretty_print()

        # Iterate atom evaluation until entire program is evaluated.
        while len(self.state["exec"]) > 0:
            # Stopping conditions
            if self.steps > push_config.step_limit:
                self.status = PushInterpreterStatus.step_limit_exceeded
                break
            if time.time() > stop_time:
                self.status = PushInterpreterStatus.runtime_limit_exceeded
                break

            # Next atom in the program to evaluate.
            next_atom = self.state["exec"].pop()

            if print_trace:
                start = time.time()
                print("\nCurrent Atom: " + str(next_atom))

            # Evaluate atom.
            old_size = self.state.size()
            self.evaluate_atom(next_atom, push_config)
            if self.state.size() > old_size + push_config.growth_cap:
                self.status = PushInterpreterStatus.growth_cap_exceeded
                break

            if print_trace:
                duration = time.time() - start
                print("Current State (step {step}):".format(step=self.steps))
                self.state.pretty_print()
                print("Step duration:", duration)
            self.steps += 1

        if print_trace:
            print("Finished program evaluation.")

        return self.state.observe_stacks(program.signature.output_stacks)
