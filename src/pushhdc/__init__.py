"""PushHDC: evolving HDC encoders with PushGP."""

from pushhdc.hypervector import HyperVector
from pushhdc.push_hdc import (
    HyperVectorPushType,
    hdc_instruction_set,
    hdc_type_library,
)

__all__ = [
    "HyperVector",
    "HyperVectorPushType",
    "hdc_instruction_set",
    "hdc_type_library",
]


def hello() -> str:
    return "Hello from pushhdc!"
