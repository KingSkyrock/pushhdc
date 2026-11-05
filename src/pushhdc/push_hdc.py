"""Push type library and instruction-set helpers for hypervectors."""

from __future__ import annotations

import numpy as np

from pushhdc.hypervector import HyperVector
from pushhdc.instructions import hdc_instructions
from pushhdc.pyshgp.push.instruction_set import InstructionSet
from pushhdc.pyshgp.push.type_library import PushTypeLibrary
from pushhdc.pyshgp.push.types import PushFloat, PushInt, PushType


class HyperVectorType(PushType):
    """Push type for bipolar hypervectors."""

    def __init__(self) -> None:
        super().__init__("hypervector", (HyperVector,))

    def coerce(self, value: object) -> HyperVector:
        if isinstance(value, HyperVector):
            return value
        if isinstance(value, np.ndarray):
            return HyperVector(data=value)
        raise TypeError(f"Cannot coerce {type(value)!r} to HyperVector")


HyperVectorPushType = HyperVectorType()


def hdc_type_library(*, register_core: bool = False) -> PushTypeLibrary:
    """Build a type library that includes the hypervector stack.

    By default only int, float, and hypervector are registered (plus the
    always-present code/exec stacks). Pass ``register_core=True`` to keep
    the full core scalar/vector set alongside hypervector.
    """
    library = PushTypeLibrary(register_core=register_core)
    if not register_core:
        library.register(PushInt).register(PushFloat)
    return library.register(HyperVectorPushType)


def hdc_instruction_set(*, register_core: bool = True) -> InstructionSet:
    """Instruction set with HDC ops and matching core stack generics."""
    type_library = hdc_type_library(register_core=False)
    return (
        InstructionSet(type_library=type_library, register_core=register_core)
        .register_list(hdc_instructions())
    )
