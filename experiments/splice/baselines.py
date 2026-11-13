"""Baselines for splice junctions, to compare evolved encoders against.

Run with `uv run --with scikit-learn python experiments/splice/baselines.py`.
"""

from collections.abc import Callable
from functools import partial

import numpy as np
from dataset import BASES, CLASSES, WINDOWS, load, split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pushhdc.evaluation import centroid_accuracy

DIMENSIONS = (64, 256, 1024, 4096, 10000)
SEEDS = range(5)


# one random hypervector of -1s and 1s per symbol. int8 keeps 10000-dimensional
# arrays small, and sums widen to int64 on their own
def random_hvs(n: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, dimension))


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def bundle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def permute(hv: np.ndarray, times: int = 1) -> np.ndarray:
    return np.roll(hv, times, axis=-1)


# the hypervectors of each window's first, second and third base, each shaped
# (sequences, 58 windows, dimension). summing over axis 1 bundles the windows
def window_bases(
    sequences: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bases = values[sequences[:, WINDOWS]]
    return bases[:, :, 0], bases[:, :, 1], bases[:, :, 2]


# binds each base's position to its value and bundles the 60 results
def record(
    sequences: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    values = random_hvs(len(BASES), dimension, rng)
    positions = random_hvs(sequences.shape[1], dimension, rng)
    return bind(positions, values[sequences]).sum(axis=1)


# the next three are hand-written stand-ins for evolved programs. each runs the
# same few ops on every window and bundles the results. this one ignores order,
# so it only counts how often each base shows up
def bundle_each_window(
    sequences: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    a, b, c = window_bases(sequences, random_hvs(len(BASES), dimension, rng))
    return bundle(bundle(a, b), c).sum(axis=1)


# an n-gram encoding. permuting by place in the window keeps the order of its
# bases, but not where the window is in the sequence
def bind_each_window_in_order(
    sequences: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    a, b, c = window_bases(sequences, random_hvs(len(BASES), dimension, rng))
    return bind(bind(a, permute(b)), permute(c, 2)).sum(axis=1)


# like `bind_each_window_in_order`, but also binds each window to where it
# starts. the encoders above that ignore position do about as well as always
# guessing "neither"
def bind_each_window_to_its_start(
    sequences: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    values = random_hvs(len(BASES), dimension, rng)
    starts = random_hvs(len(WINDOWS), dimension, rng)
    a, b, c = window_bases(sequences, values)
    return bind(starts, bind(bind(a, permute(b)), permute(c, 2))).sum(axis=1)


ENCODERS = (
    record,
    bundle_each_window,
    bind_each_window_in_order,
    bind_each_window_to_its_start,
)


# encoding all 3190 sequences at once takes gigabytes at high dimensions, so
# this encodes a few hundred at a time. every part starts from the same seed,
# so they all draw the same hvs
def in_parts(
    encode: Callable[[np.ndarray, int, np.random.Generator], np.ndarray],
    sequences: np.ndarray,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seed = rng.integers(2**32)
    return np.concatenate(
        [
            encode(part, dimension, np.random.default_rng(seed))
            for part in np.array_split(sequences, 16)
        ]
    )


# NysHD (Zhao et al., AAAI 2025), the encoder we compare against. it turns a
# similarity between samples, called a kernel, into an hv by comparing each
# sequence with 300 training sequences, called landmarks. the projection uses
# how the landmarks compare with each other, so dot products between hvs follow
# the kernel. the paper's kernel ignores position, so this uses one that
# doesn't
def nyshd(
    sequences: np.ndarray,
    train: np.ndarray,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    landmarks = sequences[rng.choice(train, size=300, replace=False)]
    values, vectors = np.linalg.eigh(weighted_degree(landmarks, landmarks))
    values[values <= 0] = 1e-15
    projection = rng.uniform(-1, 1, size=(dimension, len(landmarks)))
    projection /= np.linalg.norm(projection, axis=1, keepdims=True)
    encoding = projection @ np.diag(values**-0.5) @ vectors.T
    return np.sign(weighted_degree(sequences, landmarks) @ encoding.T)


# the weighted degree kernel (Ratsch and Sonnenburg, 2004). two sequences are
# similar when they share runs of up to `longest` bases at the same place, with
# shorter runs weighted more. we didn't tune `longest`
def weighted_degree(
    a: np.ndarray, b: np.ndarray, longest: int = 10
) -> np.ndarray:
    same = a[:, None] == b[None]
    runs = same
    similarity = np.zeros((len(a), len(b)))
    for length in range(1, longest + 1):
        weight = 2 * (longest - length + 1) / (longest * (longest + 1))
        similarity += weight * runs.sum(axis=-1)
        # a run one base longer needs this run and the base after it to match
        runs = runs[..., :-1] & same[..., length:]
    return similarity


def main() -> None:
    sequences, labels = load()
    # nothing here gets tuned, so a validation split isn't needed
    train, test = split(len(labels))

    def row(name: str, accuracies: list[float]) -> None:
        print(f"{name:32}" + "".join(f"{a:8.3f}" for a in accuracies))

    print("test accuracy")
    # most sequences are neither boundary, so always guessing that is the bar
    # to beat
    row("always neither", [(labels[test] == CLASSES.index("N")).mean()])
    one_hot = np.eye(len(BASES))[sequences].reshape(len(sequences), -1)
    models = {
        "logistic regression": LogisticRegression(max_iter=1000),
        "random forest": RandomForestClassifier(random_state=0),
    }
    for name, model in models.items():
        model.fit(one_hot[train], labels[train])
        row(name, [model.score(one_hot[test], labels[test])])

    # each seed draws different hypervectors, so hdc rows average over seeds
    print(f"\n{'hdc, by dimension':32}" + "".join(f"{d:8}" for d in DIMENSIONS))
    encoders = [
        (encode.__name__, partial(in_parts, encode, sequences))
        for encode in ENCODERS
    ]
    # NysHD picks its landmarks from train, so it needs to know the split
    encoders.append(("nyshd", partial(nyshd, sequences, train)))
    for name, encode in encoders:
        accuracies = []
        for dimension in DIMENSIONS:
            runs = []
            for seed in SEEDS:
                hvs = encode(dimension, np.random.default_rng(seed))
                runs.append(centroid_accuracy(hvs, labels, train, test))
            accuracies.append(float(np.mean(runs)))
        row(name, accuracies)


if __name__ == "__main__":
    main()
