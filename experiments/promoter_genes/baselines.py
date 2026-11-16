"""Baselines for promoter gene sequences, to compare evolved encoders against.

Run with `uv run --with scikit-learn python experiments/promoter_genes/baselines.py`.
"""

import numpy as np
from dataset import load, split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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


def centroid_accuracy(
    hvs: np.ndarray, is_promoter: np.ndarray, train: np.ndarray, test: np.ndarray
) -> float:
    prototypes = np.stack(
        [
            hvs[train][is_promoter[train] == label].sum(axis=0)
            for label in (False, True)
        ]
    ).astype(float)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    guesses = (hvs[test] @ prototypes.T).argmax(axis=1).astype(bool)
    return float((guesses == is_promoter[test]).mean())


def main() -> None:
    sequences, is_promoter = load()
    train, _, test = split(is_promoter, seed=0)

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
    for encode in ENCODERS:
        accuracies = []
        for dimension in DIMENSIONS:
            runs = []
            for seed in SEEDS:
                hvs = encode(sequences, dimension, np.random.default_rng(seed))
                runs.append(centroid_accuracy(hvs, is_promoter, train, test))
            accuracies.append(float(np.mean(runs)))
        row(encode.__name__, accuracies)


if __name__ == "__main__":
    main()