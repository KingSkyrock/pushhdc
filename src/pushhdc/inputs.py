"""The inputs of an evaluator, and how they get cut for each sample length.

An input holds a hypervector for each unit of each sample, shaped (samples,
units, dimension). It can be an array, or a `Lookup` that makes the hvs of the
units only when the evaluator cuts it. Samples can have different numbers of
units. A program runs on the samples of one length together, so the evaluator
cuts its inputs into one part for each length.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from pushhdc.hypervector import HyperVector

# samples of one length, which a program runs on together. the array says
# where each of them goes in the output, and the hvs are their inputs, cut to
# that length
type SameLength = tuple[np.ndarray, list[HyperVector]]


class Lookup:
    """an input given as one hv for each symbol and the symbol at each unit,
    so nobody has to build the padded (samples, units, dimension) array.

    unit u of sample i is `hvs[symbols[i, u]]`. `hvs` has one row for each
    symbol, and `symbols` is shaped (samples, units), or (1, units) for an
    input that every sample shares. the evaluator makes each unit's hv when it
    cuts its inputs, and with `lengths` it makes only the units within each
    length. the padded array holds a copy of a symbol's hv at each unit up to
    the longest sample, so it can be much bigger than the hvs and symbols.
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

    # the evaluator reads an input in cuts of samples and units, so only the
    # hvs of those units get made
    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self.hvs[self.symbols[key]])


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
    cut to that length. an input that all samples share keeps its one row. the
    array of each length says where its samples are in `rows`."""
    out = []
    for length in np.unique(lengths[rows]):
        where = np.flatnonzero(lengths[rows] == length)
        cut = [
            batch[:, :length]
            if len(batch) == 1
            else batch[rows[where], :length]
            for batch in inputs
        ]
        out.append((where, [HyperVector(batch) for batch in cut]))
    return out
