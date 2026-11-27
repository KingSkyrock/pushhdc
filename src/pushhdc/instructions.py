"""Push instructions for hypervector operations.

Implemented: HDC_BIND, HDC_BUNDLE, HDC_PERMUTE, HDC_COSINE_SIM
(as hv_bind, hv_bundle, hv_permute, hv_cosine_sim).

TODO: later candidates
- HDC_RANDOM_PROJECTION
- HDC_QUANTIZE
- HDC_NORMALIZE
"""

from __future__ import annotations

from pushhdc.hypervector import HyperVector
from pushhdc.pyshgp.push.instruction import (
    Instruction,
    SimpleInstruction,
)


def _hv_bind(a: HyperVector, b: HyperVector) -> tuple[HyperVector]:
    return (a.bind(b),)


def _hv_bundle(a: HyperVector, b: HyperVector) -> tuple[HyperVector]:
    return (a.bundle(b),)


def _hv_permute(hv: HyperVector, shift: int) -> tuple[HyperVector]:
    return (hv.permute(shift=int(shift)),)


def _hv_cosine_sim(a: HyperVector, b: HyperVector) -> tuple[float]:
    return (a.cosine_similarity(b),)


def hdc_instructions() -> list[Instruction]:
    """Core hypervector instructions for the Push interpreter."""
    return [
        SimpleInstruction(
            "hv_bind",
            _hv_bind,
            ["hypervector", "hypervector"],
            ["hypervector"],
            0,
            docstring="Bind (element-wise multiply) the top two hypervectors.",
        ),
        SimpleInstruction(
            "hv_bundle",
            _hv_bundle,
            ["hypervector", "hypervector"],
            ["hypervector"],
            0,
            docstring="Bundle (add) the top two hypervectors.",
        ),
        SimpleInstruction(
            "hv_permute",
            _hv_permute,
            ["hypervector", "int"],
            ["hypervector"],
            0,
            docstring="Cyclically permute the top hypervector by the top int.",
        ),
        SimpleInstruction(
            "hv_cosine_sim",
            _hv_cosine_sim,
            ["hypervector", "hypervector"],
            ["float"],
            0,
            docstring="Push the cosine similarity of the top two hypervectors.",
        ),
    ]
