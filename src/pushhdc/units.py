"""Ops that let evolution decide how the units of a sample group together.

With these ops, evolution picks the n-gram length, the gaps between the units
in a group and the region that the encoder looks at, so a person does not have
to. `hv_shift_units` lets a program combine each unit with the unit a set
number of places after it, and `hv_keep_units` lets it keep one stretch of
units and set the rest to zero. Both only make sense when the units are in
order, as the positions of a sequence are.
"""

import numpy as np

from pushhdc.hypervector import HyperVector
from pushhdc.pyshgp.push.instruction import Instruction, SimpleInstruction


# a unit near the end has no unit `k` units after it. that unit becomes
# zeros, so the bundle does not include it. the shift does not wrap around,
# because that would join the end of a sequence to its start
def _shift_units(hv: HyperVector, k: int) -> tuple[HyperVector]:
    data = np.zeros_like(hv.data)
    units = data.shape[-2]
    data[..., : max(units - k, 0), :] = hv.data[..., k:, :]
    return (HyperVector(data),)


# the ints can come in either order, because Push gives the top int first and
# a program can easily push them the other way round. the range runs from the
# smaller int to the larger one with both ends included, so it always holds
# at least one unit
def _keep_units(hv: HyperVector, a: int, b: int) -> tuple[HyperVector]:
    data = np.zeros_like(hv.data)
    low, high = min(a, b), max(a, b) + 1
    data[..., low:high, :] = hv.data[..., low:high, :]
    return (HyperVector(data),)


def unit_instructions() -> list[Instruction]:
    """the ops for programs whose units are in order."""
    return [
        SimpleInstruction(
            "hv_shift_units",
            _shift_units,
            ["hypervector", "int"],
            ["hypervector"],
            0,
            docstring="Replace each unit with the unit n units after it, or "
            "with zeros if there is none. The top int is n.",
        ),
        SimpleInstruction(
            "hv_keep_units",
            _keep_units,
            ["hypervector", "int", "int"],
            ["hypervector"],
            0,
            docstring="Keep the units from one top int to the other, ends "
            "included, and set the other units to zero.",
        ),
    ]
