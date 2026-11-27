"""Baselines for UCI HAR from the raw sensor signals, with the same levels,
hvs and classifier as the evolved encoders.

Run with
`uv run python experiments/human_activity_recognition/signal_baselines.py`.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from signals import channel_hvs, load_signals, readings

from pushhdc.evaluation import centroid_accuracy

# every experiment shares `protocol.py`, which is in `experiments/`
sys.path.append(str(Path(__file__).parents[1]))
from protocol import REPORT_CODEBOOKS

DIMENSIONS = (256, 1024)
# the n-grams below go up to runs of this many time steps
LONGEST_N = 4
FOLDS = 5
# the encoders make arrays with a number for each time step of each window in
# each dimension, so they encode a part of the windows at a time, with about
# this many numbers in each part
PART = 10_000_000


def encodings(
    levels: list[np.ndarray],
    each_channel: list[np.ndarray],
    step_hvs: np.ndarray,
) -> dict[str, np.ndarray]:
    """the hv of each window from each encoder, by name."""
    windows, steps = levels[0].shape
    part = max(1, PART // (steps * step_hvs.shape[1]))
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, windows, part):
        # the sum of the channels' level hvs at each time step
        sums = sum(
            (
                hvs[level[start : start + part]].astype(np.int64)
                for hvs, level in zip(each_channel, levels, strict=True)
            ),
            start=np.zeros(1, dtype=np.int64),
        )
        # every value fits in int32, which halves the memory
        for name, hvs in _encode(sums, step_hvs).items():
            parts[name].append(hvs.astype(np.int32))
    return {name: np.concatenate(each) for name, each in parts.items()}


# "record" binds the sum at each time step to the hv of the step. the others
# are n-grams over time of the sum at each step, or of its sign as in the
# usual HDC encoder for many sensors. with n = 1 they count the sums
# wherever they are
def _encode(sums: np.ndarray, step_hvs: np.ndarray) -> dict[str, np.ndarray]:
    out = {"record": (sums * step_hvs).sum(axis=1)}
    # 9 channels give an odd sum, so the sign is never 0
    for kind, each_step in (("sum", sums), ("sign", np.sign(sums))):
        for n in range(1, LONGEST_N + 1):
            out[f"{kind} {n}"] = _ngrams(each_step, n)
    return out


# each run of n time steps, with each step permuted by its place in the run
# and the steps bound together, summed over the runs
def _ngrams(each_step: np.ndarray, n: int) -> np.ndarray:
    runs = each_step.shape[1] - n + 1
    bound = each_step[:, :runs].copy()
    for place in range(1, n):
        bound *= np.roll(each_step[:, place : place + runs], place, axis=-1)
    return bound.sum(axis=1)


# folds over the train windows that keep the windows of each person together,
# as the evaluator does with `sources`. each pair is the train and the
# held-out windows
def _folds_by_person(
    people: np.ndarray, train: np.ndarray, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    shuffled = np.random.default_rng(seed).permutation(np.unique(people[train]))
    out = []
    for part in np.array_split(shuffled, FOLDS):
        held = np.isin(people[train], part)
        out.append((train[~held], train[held]))
    return out


def _cross_validated(
    hvs: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    accuracies = [centroid_accuracy(hvs, labels, a, b) for a, b in folds]
    return float(np.average(accuracies, weights=[len(b) for _, b in folds]))


def main() -> None:
    signals, labels, people, train, test = load_signals()
    levels = readings(signals, train)
    steps = signals.shape[1]
    # test accuracy over the report codebooks, by row and dimension
    rows: dict[str, list[float]] = defaultdict(list)
    picked: dict[str, list[list[int]]] = defaultdict(list)
    for dimension in DIMENSIONS:
        tested: dict[str, list[float]] = defaultdict(list)
        for kind in ("sum", "sign"):
            picked[kind].append([])
        for codebook in REPORT_CODEBOOKS:
            print(f"d={dimension} codebook {codebook}...", file=sys.stderr)
            hvs = encodings(levels, *channel_hvs(codebook, dimension, steps))
            for name, encoded in hvs.items():
                tested[name].append(
                    centroid_accuracy(encoded, labels, train, test)
                )
            # the automatic rows pick n by cross-validation on train only
            folds = _folds_by_person(people, train, codebook)
            for kind in ("sum", "sign"):
                cv = [
                    _cross_validated(hvs[f"{kind} {n}"], labels, folds)
                    for n in range(1, LONGEST_N + 1)
                ]
                n = int(np.argmax(cv)) + 1
                picked[kind][-1].append(n)
                tested[f"{kind} picked"].append(tested[f"{kind} {n}"][-1])
            # so that the hvs of two codebooks are never in memory together
            del hvs
        for name, accuracies in tested.items():
            rows[name].append(float(np.mean(accuracies)))

    def row(label: str, accuracies: list[float]) -> None:
        print(f"{label:32}" + "".join(f"{a:8.3f}" for a in accuracies))

    print("test accuracy, nearest centroid, mean over the report codebooks")
    most_common = np.bincount(labels[train]).argmax()
    row("majority class", [float(np.mean(labels[test] == most_common))])
    print(f"{'by dimension':32}" + "".join(f"{d:8}" for d in DIMENSIONS))
    row("bag of channel sums", rows["sum 1"])
    row("record", rows["record"])
    for kind in ("sum", "sign"):
        for n in range(2, LONGEST_N + 1):
            row(f"{n}-grams of channel {kind}s", rows[f"{kind} {n}"])
        row(f"n-grams of channel {kind}s, n by cv", rows[f"{kind} picked"])
    for kind in ("sum", "sign"):
        print(f"n picked for channel {kind}s: {picked[kind]}")


if __name__ == "__main__":
    main()
