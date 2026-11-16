"""Loads the UCI Molecular Biology (Splice-junction Gene Sequences) dataset and splits it."""

import io
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data"
PATH = Path(__file__).parents[2] / "data" / "splice_junctions" / "splice.data"

# Standard nucleotides mapped to 0, 1, 2, 3 (ambiguous characters map to index 4)
NUCLEOTIDES = "acgtn"


def load() -> tuple[np.ndarray, np.ndarray]:
    """Loads splice sequences and their categorical labels (ei, ie, n).
    
    Returns:
        sequences: NumPy array of shape (N, 60) with integer-encoded nucleotides.
        labels: String array of classification labels.
    """
    if not PATH.exists():
        _download()
    
    lines = [line.strip() for line in PATH.read_text().splitlines() if line.strip()]
    sequences = []
    labels = []
    
    for line in lines:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        label = parts[0].strip().lower()  # 'ei', 'ie', or 'n'
        seq_str = parts[2].strip().replace(",", "").lower()
        
        if len(seq_str) != 60:
            continue
            
        encoded_seq = [NUCLEOTIDES.index(n) if n in NUCLEOTIDES else 4 for n in seq_str]
        if len(encoded_seq) == 60:
            sequences.append(encoded_seq)
            labels.append(label)
            
    return np.array(sequences, dtype=np.int64), np.array(labels, dtype=object)


def split(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train, validation and test indices (60/20/20 mix) across classes."""
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
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