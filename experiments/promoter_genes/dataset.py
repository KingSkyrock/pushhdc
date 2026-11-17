"""Loads the UCI Molecular Biology (Promoter Gene Sequences) dataset and splits it."""

import urllib.request
from pathlib import Path

import numpy as np

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
PATH = Path(__file__).parents[2] / "data" / "promoters" / "promoters.data"

# Nucleotide mapping: 'a', 'c', 'g', 't' mapped to integers 0, 1, 2, 3
NUCLEOTIDES = "acgt"


def load() -> tuple[np.ndarray, np.ndarray]:
    """Loads all 106 promoter sequences and their binary labels (+ / -).
    Returns:
        sequences: NumPy array of shape (106, 57) with integer-encoded nucleotides.
        is_promoter: Boolean NumPy array indicating whether each sequence is a promoter.
    """
    if not PATH.exists():
        _download()

    lines = [
        line.strip() for line in PATH.read_text().splitlines() if line.strip()
    ]
    sequences = []
    is_promoter = []

    for line in lines:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        label = parts[0].strip()  # '+' or '-'
        # parts[2] contains the 57-long nucleotide sequence, stripping whitespace/commas
        seq_str = parts[2].strip().replace(",", "").lower()

        if len(seq_str) != 57:
            continue

        encoded_seq = [
            NUCLEOTIDES.index(n) for n in seq_str if n in NUCLEOTIDES
        ]
        if len(encoded_seq) == 57:
            sequences.append(encoded_seq)
            is_promoter.append(label == "+")

    return np.array(sequences, dtype=np.int64), np.array(
        is_promoter, dtype=bool
    )


def split(
    is_promoter: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train, validation and test indices (60/20/20 mix)."""
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for label in (True, False):
        indices = rng.permutation(np.flatnonzero(is_promoter == label))
        train_end = round(0.6 * len(indices))
        validation_end = round(0.8 * len(indices))
        train.extend(indices[:train_end])
        validation.extend(indices[train_end:validation_end])
        test.extend(indices[validation_end:])
    return np.sort(train), np.sort(validation), np.sort(test)


def _download() -> None:
    with urllib.request.urlopen(URL) as response:
        content = response.read()
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_bytes(content)
