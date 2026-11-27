"""The inputs of an evaluator, and how they get cut for each sample length.

An input holds a hypervector for each unit of each sample, shaped (samples,
units, dimension). It can be an array, or a `Lookup` that makes the hvs of the
units only while a program reads them. Samples can have different numbers of
units. A program runs on the samples of one length together, so the evaluator
cuts its inputs into one part for each length.
"""

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from pushhdc.hypervector import HyperVector


class SameLength(NamedTuple):
    """samples of one length, which a program runs on together. `places` says
    where each of them is in the `rows` that `by_length` cut them from, and
    `inputs` are their inputs, cut to that length. a `Lookup` stays a
    `Lookup`, so the hvs of its units get made only while a program reads
    them."""

    places: np.ndarray
    inputs: "list[HyperVector | Lookup]"


class Lookup:
    """an input given as one hv for each symbol and the symbol at each unit,
    so nobody has to build the padded (samples, units, dimension) array.

    unit u of sample i is `hvs[symbols[i, u]]`. `hvs` has one row for each
    symbol, and `symbols` is shaped (samples, units), or (1, units) for an
    input that every sample shares. a cut of a `Lookup` is a `Lookup` too, and
    the evaluator makes the hv of each unit only while a program reads the
    input, so the evaluator and its copies in worker processes stay small.
    with `lengths` it makes only the units within each length. the padded
    array holds a copy of a symbol's hv at each unit up to the longest
    sample, so it can be much bigger than the hvs and symbols.
    the symbols past each length are never read, but each must still pick
    an hv, for example 0.
    """

    # the evaluator reads an input as it reads an array, by `ndim`, `shape`,
    # `len` and cuts
    ndim = 3

    def __init__(self, hvs: np.ndarray, symbols: np.ndarray) -> None:
        hvs, symbols = np.asarray(hvs), np.asarray(symbols)
        if hvs.ndim != 2 or symbols.ndim != 2:
            raise ValueError(
                "hvs must have one row for each symbol, and symbols must be "
                "shaped (samples, units)"
            )
        # numpy would take -1 as the last hv with no error
        if not np.issubdtype(symbols.dtype, np.integer) or (
            symbols.size > 0
            and (symbols.min() < 0 or symbols.max() >= len(hvs))
        ):
            raise ValueError(
                f"symbols must pick one of the {len(hvs)} hvs, so they must be "
                f"whole numbers from 0 to {len(hvs) - 1}"
            )
        self.hvs = hvs
        self.symbols = symbols
        self.shape = (*symbols.shape, hvs.shape[1])

    def __len__(self) -> int:
        return len(self.symbols)

    # the evaluator reads an input in cuts of samples and units. a cut keeps
    # the same hvs and only the symbols of those units
    def __getitem__(self, key: Any) -> "Lookup":
        return Lookup(self.hvs, self.symbols[key])

    def array(self) -> np.ndarray:
        """the hv of each unit, shaped (samples, units, dimension)."""
        return np.asarray(self.hvs[self.symbols])


def checked_lengths(
    inputs: Sequence[np.ndarray | Lookup],
    lengths: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    """the number of units of each sample in `inputs`, as an array. it first
    checks that each input is shaped (samples, units, dimension). with no
    `lengths`, every sample has all the units."""
    if any(batch.ndim != 3 for batch in inputs):
        raise ValueError("inputs must be shaped (samples, units, dimension)")
    count = max(len(batch) for batch in inputs)
    units = inputs[0].shape[1]
    lengths = np.full(count, units) if lengths is None else np.asarray(lengths)
    if (
        not np.issubdtype(lengths.dtype, np.integer)
        or lengths.shape != (count,)
        or lengths.min() < 0
        or lengths.max() > units
    ):
        raise ValueError(
            f"lengths must give each of the {count} samples in the inputs "
            f"a whole number of units from 0 to {units}"
        )
    return lengths


def by_length(
    inputs: Sequence[np.ndarray | Lookup], lengths: np.ndarray, rows: np.ndarray
) -> list[SameLength]:
    """the samples in `rows` put together by length, with each of their inputs
    cut to that length. an input that all samples share keeps its one row."""
    out = []
    for length in np.unique(lengths[rows]):
        places = np.flatnonzero(lengths[rows] == length)
        cut = [
            batch[:, :length]
            if len(batch) == 1
            else batch[rows[places], :length]
            for batch in inputs
        ]
        out.append(
            SameLength(
                places,
                [
                    batch if isinstance(batch, Lookup) else HyperVector(batch)
                    for batch in cut
                ],
            )
        )
    return out
