"""The ops that an encoder program can use, what each op costs, and how
evolution makes random genes from them."""

import numpy as np

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


def largest_int(seen_lengths: np.ndarray) -> int:
    """the largest int that a new gene gets with `unit_ops`, which is the place
    of the last unit in the longest sample. `seen_lengths` gives the number of
    units of each sample that evolution sees."""
    return int(seen_lengths.max()) - 1


def encoder_spawner(
    instructions: InstructionSet,
    arity: int,
    *,
    unit_ops: bool,
    seen_lengths: np.ndarray,
) -> GeneSpawner:
    """makes random genes from `instructions` for a program with `arity`
    inputs. with `unit_ops`, the genes also hold random ints from 0 to
    `largest_int`. `seen_lengths` gives the number of units of each sample
    that evolution sees."""
    if unit_ops:
        # the unit operations use ints for the shift distance and for the
        # ends of a range, and `hv_permute` uses the same ints. half of the
        # ints go from 0 to the last unit of the longest sample, so a program
        # can reach the far end of long samples or find where samples end.
        # the other half are places in random samples. then short shifts and
        # ranges that fit in a typical sample are common, also when most
        # samples are much shorter than the longest
        largest = largest_int(seen_lengths)
        # an empty sample has no place, so it is left out
        nonempty = seen_lengths[seen_lengths > 0]
        # when each sample is empty or as long as the longest, both halves
        # draw from 0 to the last unit. then the draw does not choose a half.
        # so a seed gives the same ints as plain draws from 0 to the last unit
        one_length = bool((nonempty == largest + 1).all())

        def draw(rng: np.random.Generator) -> int:
            if one_length or rng.random() < 0.5:
                return int(rng.integers(0, largest + 1))
            return int(rng.integers(0, nonempty[rng.integers(len(nonempty))]))

        return GeneSpawner(
            n_inputs=arity,
            instruction_set=instructions,
            literals=[],
            erc_generators=[draw],
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
