"""Loads the UCI Molecular Biology (Promoter Gene Sequences) dataset and splits it."""

import math
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

    # NysHD's code keeps every line, and its split depends on the order, so a
    # skipped line would change which sequences are in test
    if len(sequences) != 106:
        raise ValueError(f"found {len(sequences)} sequences, not 106")
    return np.array(sequences, dtype=np.int64), np.array(
        is_promoter, dtype=bool
    )


def split(count: int) -> tuple[np.ndarray, np.ndarray]:
    """train and test indices. this is the split in NysHD's code (Zhao et
    al., AAAI 2025), so the test sequences are the same as theirs."""
    # matches scikit-learn's `train_test_split(test_size=0.2, random_state=42)`
    # on the sequences in file order, as NysHD's code loads them, so
    # `evolve.py` does not need scikit-learn
    shuffled = np.random.RandomState(42).permutation(count)
    test_count = math.ceil(0.2 * count)
    return np.sort(shuffled[test_count:]), np.sort(shuffled[:test_count])


def _download() -> None:
    with urllib.request.urlopen(URL) as response:
        content = response.read()
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_bytes(content)
