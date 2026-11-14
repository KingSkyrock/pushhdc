"""Push instructions for hypervector operations.

Implemented: HDC_BIND, HDC_BUNDLE, HDC_PERMUTE, HDC_COSINE_SIM
(as hv_bind, hv_bundle, hv_permute, hv_cosine_sim).

TODO: later candidates from the original stub list
- HDC_RANDOM_PROJECTION
- HDC_QUANTIZE
- HDC_NORMALIZE
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np

from pushhdc.hypervector import HyperVector
from pushhdc.pyshgp.push.config import PushConfig
from pushhdc.pyshgp.push.instruction import (
    Instruction,
    SimpleInstruction,
    _check_is_seq,
)
from pushhdc.pyshgp.push.state import PushState
from pushhdc.pyshgp.push.type_library import RESERVED_PSEUDO_STACKS
from pushhdc.pyshgp.utils import Token


class RngAwareInstruction(Instruction):  # type: ignore[misc]
    """Like SimpleInstruction, but passes ``push_config.rng`` into ``f``.

    ``f`` must accept the popped stack args plus a keyword-only ``rng``.
    """

    __slots__ = [
        "code_blocks",
        "cost",
        "docstring",
        "f",
        "input_stacks",
        "name",
        "output_stacks",
    ]

    def __init__(
        self,
        name: str,
        f: Callable[..., Any],
        input_stacks: Sequence[str],
        output_stacks: Sequence[str],
        code_blocks: int,
        docstring: str = "Write me!",
        cost: int = 0,
    ) -> None:
        super().__init__(name, code_blocks, docstring, cost)
        self.f = f
        self.input_stacks = input_stacks
        self.output_stacks = output_stacks

    def evaluate(
        self, push_state: PushState, push_config: PushConfig | None = None
    ) -> PushState:
        args = push_state.observe_stacks(self.input_stacks)
        if Token.no_stack_item in args:
            return push_state

        config = (
            push_config if push_config is not None else push_state.push_config
        )
        rng = config.rng
        if rng is None:
            rng = np.random.default_rng()

        result = self.f(*args, rng=rng)
        if result is Token.revert:
            return push_state
        _check_is_seq(result, self)

        push_state.pop_from_stacks(self.input_stacks)
        push_state.push_to_stacks(result, self.output_stacks)
        return push_state

    def required_stacks(self) -> set[str]:
        stacks = set(self.input_stacks) | set(self.output_stacks)
        return stacks - cast(set[str], set(RESERVED_PSEUDO_STACKS))


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
