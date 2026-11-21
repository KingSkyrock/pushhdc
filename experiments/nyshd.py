"""NysHD, the encoder the string experiments compare against.

NysHD (Zhao et al., AAAI 2025) turns a similarity between samples, called a
kernel, into an hv by comparing each sample with training samples, called
landmarks. The projection uses how the landmarks compare with each other, so
dot products between hvs follow the kernel. Its string kernel is the gappy
pair kernel. This runs NysHD's own code for that kernel with no changes, from
a clone of their code in `nyshd/` at the top of this repo. Their code is at
https://github.com/QuanlingZhao/AAAI-25-Bridging-the-Gap-Between-HDC-and-Kernel-Methods-via-the-Nystrom-Method

The paper gives one kernel setting for each dataset. Here NysHD gets the
settings around it and keeps the one that cross-validation on train likes
best, so a person does not choose its kernel, the same as for evolution.
"""

import ast
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, vstack

from pushhdc.evaluation import centroid_accuracy

type Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]
# (k, g, apart). the gappy pair kernel counts pairs of k-mers up to g letters
# apart, or k-mers alone when g is 0, and with `apart` it counts each gap
# size apart
type Setting = tuple[int, int, bool]

CODE = (
    Path(__file__).parents[1]
    / "nyshd"
    / "Framework"
    / "NysHD"
    / "gen_kernel_matrix.py"
)
# their code makes a dense row for each sequence, so the features get made a
# few sequences at a time, with about this many numbers in each part
PART = 25_000_000


def settings(longest_k: int) -> list[Setting]:
    """the settings NysHD can pick from, with k-mers up to `longest_k`."""
    return [(k, 0, False) for k in range(1, longest_k + 1)] + [
        (k, g, apart)
        for k in range(1, longest_k + 1)
        for g in (1, 2, 3)
        for apart in (False, True)
    ]


def kernels(
    strings: list[str], alphabet: int, menu: list[Setting]
) -> dict[Setting, Kernel]:
    """NysHD's kernel for each setting on the menu, over sample numbers.
    `alphabet` is the `t` of their code, 0 for DNA, 2 for proteins and 3 for
    text."""
    space = _their_code()
    letters = len(space["alphabets"][alphabet])
    out: dict[Setting, Kernel] = {}
    for k, g, apart in menu:
        width = letters ** (2 * k if g else k) * (g + 1 if g and apart else 1)
        size = max(1, PART // width)
        features = vstack(
            [
                space["gappypair_kernel"](
                    strings[start : start + size],
                    k=k,
                    g=g,
                    t=alphabet,
                    include_flanking=True,
                    gapDifferent=apart,
                    sparse=True,
                )
                for start in range(0, len(strings), size)
            ]
        ).tocsr()

        def kernel(
            a: np.ndarray, b: np.ndarray, features: Any = features
        ) -> np.ndarray:
            return np.asarray((features[a] @ features[b].T).toarray())

        out[k, g, apart] = kernel
    return out


def encode(
    kernel: Kernel,
    pool: np.ndarray,
    rows: np.ndarray,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """the NysHD hvs of the samples in `rows`, with landmarks from `pool`."""
    # NysHD's code takes 2% of train as landmarks, at least 300 and at most
    # all of train
    count = min(max(int(len(pool) * 0.02), 300), len(pool))
    landmarks = rng.choice(pool, size=count, replace=False)
    values, vectors = np.linalg.eigh(kernel(landmarks, landmarks))
    values[values <= 0] = 1e-15
    projection = rng.uniform(-1, 1, size=(dimension, len(landmarks)))
    projection /= np.linalg.norm(projection, axis=1, keepdims=True)
    encoding = projection @ np.diag(values**-0.5) @ vectors.T
    return np.sign(kernel(rows, landmarks) @ encoding.T)


def cross_validated(
    kernel: Kernel, labels: np.ndarray, train: np.ndarray, dimension: int
) -> float:
    """mean accuracy over 5 folds of `train`, with landmarks from the other
    folds. each fold uses the same seed for every kernel, so the kernels get
    the same landmarks and projection."""
    order = np.random.RandomState(0).permutation(len(train))
    accuracies = []
    for fold, held_out in enumerate(np.array_split(order, 5)):
        inner = np.setdiff1d(order, held_out)
        rng = np.random.default_rng(100 + fold)
        hvs = encode(kernel, train[inner], train, dimension, rng)
        accuracies.append(
            centroid_accuracy(hvs, labels[train], inner, held_out)
        )
    return float(np.mean(accuracies))


def automatic(
    kernels: dict[Setting, Kernel], labels: np.ndarray, train: np.ndarray
) -> tuple[
    Callable[[int, np.random.Generator], np.ndarray], Callable[[int], Setting]
]:
    """an encoder for every sample, called with a dimension and a random
    generator, and the setting it uses at each dimension. the setting is the
    one that cross-validation on `train` likes best."""

    @cache
    def setting(dimension: int) -> Setting:
        return max(
            kernels,
            key=lambda s: cross_validated(kernels[s], labels, train, dimension),
        )

    every = np.arange(len(labels))

    def encoder(dimension: int, rng: np.random.Generator) -> np.ndarray:
        return encode(kernels[setting(dimension)], train, every, dimension, rng)

    return encoder, setting


# their file also loads data for their other experiments, which needs
# packages we do not use, so only its settings and functions run
@cache
def _their_code() -> dict[str, Any]:
    tree = ast.parse(CODE.read_text())
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
    exec(compile(ast.Module(keep, type_ignores=[]), CODE, "exec"), space)
    return space
