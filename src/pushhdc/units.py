"""Operations that let evolution choose how units make groups.

The units must be in order, for example the positions of a sequence. A program
can combine each unit with a unit that comes after it. A program can also keep
only one part of the sequence. Evolution then chooses the n-gram length, the
gaps and the region. A person does not have to choose them.
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


# the two ints can be in either order. Push gives the top int first, and it is
# easy to use the wrong order. the range always contains one or more units
def _keep_units(hv: HyperVector, a: int, b: int) -> tuple[HyperVector]:
    data = np.zeros_like(hv.data)
    low, high = min(a, b), max(a, b) + 1
    data[..., low:high, :] = hv.data[..., low:high, :]
    return (HyperVector(data),)


def unit_instructions() -> list[Instruction]:
    """the operations for programs whose units are in order."""
    return [
        SimpleInstruction(
            "hv_shift_units",
            _shift_units,
            ["hypervector", "int"],
            ["hypervector"],
            0,
            docstring="Replace each unit with the unit that is n units after "
            "it. The top int is n.",
        ),
        SimpleInstruction(
            "hv_keep_units",
            _keep_units,
            ["hypervector", "int", "int"],
            ["hypervector"],
            0,
            docstring="Set to zero each unit that is not between the top two "
            "ints.",
        ),
    ]
