"""Baselines for splice junctions, to compare evolved encoders against.

Run with `uv run --with scikit-learn python experiments/splice/baselines.py`.
The NysHD row needs a clone of NysHD's code in `nyshd/` at the top of this
repo. Their code is at
https://github.com/QuanlingZhao/AAAI-25-Bridging-the-Gap-Between-HDC-and-Kernel-Methods-via-the-Nystrom-Method
"""

import ast
from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from typing import Any

import numpy as np
from dataset import BASES, CLASSES, WINDOWS, load, split
from scipy.sparse import csr_matrix, vstack
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
# starts, so the hv keeps the place of each window in the sequence
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


# encoding every sequence at once needs too much memory at high dimensions,
# so this encodes a few hundred at a time. every part starts from the same
# seed, so they all draw the same hvs
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


type Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]

NYSHD_CODE = (
    Path(__file__).parents[2]
    / "nyshd"
    / "Framework"
    / "NysHD"
    / "gen_kernel_matrix.py"
)
# NysHD's string kernel is the gappy pair kernel. it counts pairs of k-mers up
# to g bases apart, or k-mers alone when g is 0, and with `apart` it counts
# each gap size apart. the paper gives one setting for splice, (2, 1, False).
# NysHD gets the settings around it and keeps the one that cross-validation on
# train likes best, so a person does not choose its kernel, the same as for
# evolution
NYSHD_SETTINGS = [(k, 0, False) for k in (1, 2, 3)] + [
    (k, g, apart)
    for k in (1, 2, 3)
    for g in (1, 2, 3)
    for apart in (False, True)
]


# NysHD (Zhao et al., AAAI 2025), the encoder we compare against. it turns a
# similarity between samples, called a kernel, into an hv by comparing each
# sample with training samples, called landmarks. the projection uses how the
# landmarks compare with each other, so dot products between hvs follow the
# kernel. this encodes the samples in `rows`, with landmarks from `pool`
def nyshd(
    kernel: Kernel,
    pool: np.ndarray,
    rows: np.ndarray,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = max(int(len(pool) * 0.02), 300)
    landmarks = rng.choice(pool, size=count, replace=False)
    values, vectors = np.linalg.eigh(kernel(landmarks, landmarks))
    values[values <= 0] = 1e-15
    projection = rng.uniform(-1, 1, size=(dimension, len(landmarks)))
    projection /= np.linalg.norm(projection, axis=1, keepdims=True)
    encoding = projection @ np.diag(values**-0.5) @ vectors.T
    return np.sign(kernel(rows, landmarks) @ encoding.T)


# NysHD's kernel for each setting, over sample numbers. it runs their code with
# no changes. their file also loads data for their other experiments, which
# needs packages we do not use, so only its settings and functions run. their
# code makes a dense row for each sequence, so the features get made a few
# hundred sequences at a time
def nyshd_kernels(sequences: np.ndarray) -> dict[tuple[int, int, bool], Kernel]:
    tree = ast.parse(NYSHD_CODE.read_text())
    keep = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or (
            isinstance(node, ast.FunctionDef)
            and node.name != "gen_kernel_matrix"
        )
    ]
    space: dict[str, Any] = {"np": np, "csr_matrix": csr_matrix}
    code = compile(ast.Module(keep, type_ignores=[]), NYSHD_CODE, "exec")
    exec(code, space)
    strings = ["".join(BASES[base] for base in s) for s in sequences]
    kernels = {}
    for k, g, apart in NYSHD_SETTINGS:
        # the other arguments are the ones their code uses for splice
        features = vstack(
            [
                space["gappypair_kernel"](
                    strings[start : start + 200],
                    k=k,
                    g=g,
                    t=0,
                    include_flanking=True,
                    gapDifferent=apart,
                    sparse=True,
                )
                for start in range(0, len(strings), 200)
            ]
        ).tocsr()

        def kernel(
            a: np.ndarray, b: np.ndarray, features: Any = features
        ) -> np.ndarray:
            return np.asarray((features[a] @ features[b].T).toarray())

        kernels[k, g, apart] = kernel
    return kernels


# mean accuracy over folds of train, with landmarks from the other folds. each
# fold uses the same seed for every setting, so the settings get the same
# landmarks and projection
def cross_validated(
    kernel: Kernel, labels: np.ndarray, train: np.ndarray, dimension: int
) -> float:
    order = np.random.RandomState(0).permutation(len(train))
    accuracies = []
    for fold, held_out in enumerate(np.array_split(order, 5)):
        inner = np.setdiff1d(order, held_out)
        rng = np.random.default_rng(100 + fold)
        hvs = nyshd(kernel, train[inner], train, dimension, rng)
        accuracies.append(
            centroid_accuracy(hvs, labels[train], inner, held_out)
        )
    return float(np.mean(accuracies))


def main() -> None:
    sequences, labels = load()
    # nothing here gets tuned, so a validation split is not needed
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
    # NysHD picks its landmarks and its kernel setting from train, so it needs
    # to know the split
    kernels = nyshd_kernels(sequences)

    @cache
    def setting(dimension: int) -> tuple[int, int, bool]:
        return max(
            NYSHD_SETTINGS,
            key=lambda s: cross_validated(kernels[s], labels, train, dimension),
        )

    every = np.arange(len(labels))
    encoders.append(
        (
            "nyshd",
            lambda dimension, rng: nyshd(
                kernels[setting(dimension)], train, every, dimension, rng
            ),
        )
    )
    for name, encode in encoders:
        accuracies = []
        for dimension in DIMENSIONS:
            runs = []
            for seed in SEEDS:
                hvs = encode(dimension, np.random.default_rng(seed))
                runs.append(centroid_accuracy(hvs, labels, train, test))
            accuracies.append(float(np.mean(runs)))
        row(name, accuracies)
    print("\nnyshd kernel setting (k, g, apart), picked on train")
    for dimension in DIMENSIONS:
        print(f"{dimension:8}  {setting(dimension)}")


if __name__ == "__main__":
    main()
