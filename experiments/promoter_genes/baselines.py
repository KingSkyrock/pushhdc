"""Baselines for promoter gene sequences, to compare evolved encoders against.

Run with `uv run --with scikit-learn python experiments/promoter_genes/baselines.py`.
The NysHD row needs a clone of NysHD's code in `nyshd/` at the top of this
repo. Their code is at
https://github.com/QuanlingZhao/AAAI-25-Bridging-the-Gap-Between-HDC-and-Kernel-Methods-via-the-Nystrom-Method
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
from dataset import NUCLEOTIDES, load, split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pushhdc.evaluation import centroid_accuracy

# the string experiments share the NysHD baseline, which is in `experiments/`
sys.path.append(str(Path(__file__).parents[1]))
import nyshd

DIMENSIONS = (64, 256, 1024, 4096, 10000)
SEEDS = range(5)
NUM_NUCLEOTIDES = 4  # a, c, g, t


def random_hvs(n: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, dimension))


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def bundle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def permute(hv: np.ndarray, times: int = 1) -> np.ndarray:
    return np.roll(hv, times, axis=-1)


def bag_of_bases(
    sequences: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    """Unordered bundle of all nucleotides in the sequence."""
    bases = random_hvs(NUM_NUCLEOTIDES, dimension, rng)
    return bases[sequences].sum(axis=1)


def record(
    sequences: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    """Standard HDC record encoding: binds each position (0..56) to its nucleotide."""
    seq_len = sequences.shape[1]
    bases = random_hvs(NUM_NUCLEOTIDES, dimension, rng)
    positions = random_hvs(seq_len, dimension, rng)
    return bind(positions, bases[sequences]).sum(axis=1)


def _encode_kmers(
    sequences: np.ndarray, k: int, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    """Sliding k-mer encoder using cyclic permutation for base order within each window."""
    bases = random_hvs(NUM_NUCLEOTIDES, dimension, rng)
    seq_hvs = bases[sequences]  # (num_seqs, seq_len, dimension)
    seq_len = sequences.shape[1]

    kmer_hvs = seq_hvs[:, : seq_len - k + 1].copy()
    for offset in range(1, k):
        window = seq_hvs[:, offset : seq_len - k + 1 + offset]
        kmer_hvs = bind(kmer_hvs, permute(window, offset))

    return kmer_hvs.sum(axis=1)


def kmer_3(
    sequences: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    """Bundles overlapping 3-mers (codons/triplets) across the sequence."""
    return _encode_kmers(sequences, k=3, dimension=dimension, rng=rng)


def kmer_6(
    sequences: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    """Bundles overlapping 6-mers (standard prokaryotic promoter motif length)."""
    return _encode_kmers(sequences, k=6, dimension=dimension, rng=rng)


ENCODERS = (bag_of_bases, record, kmer_3, kmer_6)


def main() -> None:
    sequences, is_promoter = load()
    # nothing here gets tuned, so a validation split is not needed
    train, test = split(len(is_promoter))

    def row(name: str, accuracies: list[float]) -> None:
        print(f"{name:32}" + "".join(f"{a:8.3f}" for a in accuracies))

    print("test accuracy")
    majority_acc = max(is_promoter[test].mean(), (~is_promoter[test]).mean())
    row("majority class", [float(majority_acc)])

    one_hot = np.eye(NUM_NUCLEOTIDES)[sequences].reshape(len(sequences), -1)
    models = {
        "logistic regression": LogisticRegression(max_iter=1000),
        "random forest": RandomForestClassifier(random_state=0),
    }
    for name, model in models.items():
        model.fit(one_hot[train], is_promoter[train])
        row(name, [float(model.score(one_hot[test], is_promoter[test]))])

    print(f"\n{'hdc, by dimension':32}" + "".join(f"{d:8}" for d in DIMENSIONS))
    encoders = [
        (encode.__name__, partial(encode, sequences)) for encode in ENCODERS
    ]
    # NysHD picks its landmarks and its kernel setting from train, so it needs
    # to know the split. the paper's setting for promoter is (2, 1, False).
    # their code reads the sequences in capitals
    strings = ["".join(NUCLEOTIDES.upper()[n] for n in s) for s in sequences]
    kernels = nyshd.kernels(strings, 0, nyshd.settings(3))
    encode_nyshd, setting = nyshd.automatic(kernels, is_promoter, train)
    encoders.append(("nyshd", encode_nyshd))
    for name, encode in encoders:
        accuracies = []
        for dimension in DIMENSIONS:
            runs = []
            for seed in SEEDS:
                hvs = encode(dimension, np.random.default_rng(seed))
                runs.append(centroid_accuracy(hvs, is_promoter, train, test))
            accuracies.append(float(np.mean(runs)))
        row(name, accuracies)
    print("\nnyshd kernel setting (k, g, apart), picked on train")
    for dimension in DIMENSIONS:
        print(f"{dimension:8}  {setting(dimension)}")


if __name__ == "__main__":
    main()
