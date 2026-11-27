"""How good a genome is, in the form that scoring and evolution share."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Fitness:
    """how good a genome is. higher accuracy is better, then lower cost."""

    accuracy: float
    cost: float


type Evaluate = Callable[[Any], Fitness]
type Scored = list[tuple[Any, Fitness]]
