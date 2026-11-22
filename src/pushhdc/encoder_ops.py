"""The ops that an encoder program can use, what each op costs, and how
evolution makes random genes from them."""

from pushhdc.push_hdc import hdc_instruction_set
from pushhdc.pyshgp.gp.genome import GeneSpawner
from pushhdc.pyshgp.push.instruction_set import InstructionSet
from pushhdc.units import unit_instructions

# ops that only move data, leave units out or copy an hv have no cost.
# hardware does them with wiring and no arithmetic, and it can use the same hv
# again in place of a copy
_FREE_OPS = {"hv_permute", "hv_shift_units", "hv_keep_units", "hypervector_dup"}


def encoder_instructions(unit_ops: bool) -> InstructionSet:
    """the ops a program can use, with their costs. an encoder only needs the
    hv ops, since similarity is the classifier's job. every op that does
    arithmetic has the same cost. with `unit_ops`, it adds the ops that
    combine and keep units, and `hypervector_dup`."""
    instructions = hdc_instruction_set(register_core=False)
    instructions.unregister("hv_cosine_sim")
    if unit_ops:
        instructions.register_list(unit_instructions())
        # a program can use one result more than one time, and does not have
        # to make it again
        instructions.register_core_by_name("hypervector_dup$")
    for name, instruction in instructions.items():
        instruction.cost = 0 if name in _FREE_OPS else 1
    return instructions


def encoder_spawner(
    instructions: InstructionSet, arity: int, *, unit_ops: bool, units: int
) -> GeneSpawner:
    """makes random genes from `instructions` for a program with `arity`
    inputs. with `unit_ops`, the genes also hold random ints below `units`,
    the length of the longest sample that evolution sees."""
    if unit_ops:
        # the unit operations use ints for the shift distance and for the
        # ends of a range, and `hv_permute` uses the same ints. the ints
        # reach from the first unit to the last, so a range can start at
        # the first unit
        return GeneSpawner(
            n_inputs=arity,
            instruction_set=instructions,
            literals=[],
            erc_generators=[lambda rng: int(rng.integers(0, units))],
        )
    # hv_permute takes its shift from the int stack. shifts add up when a
    # program permutes again, so the smallest shift, the literal below,
    # can reach any shift
    return GeneSpawner(
        n_inputs=arity,
        instruction_set=instructions,
        literals=[1],
        erc_generators=[],
    )
