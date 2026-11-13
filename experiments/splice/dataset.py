"""Loads the UCI splice-junction DNA sequences and splits them."""

import io
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://archive.ics.uci.edu/static/public/69/molecular+biology+splice+junction+gene+sequences.zip"
PATH = Path(__file__).parents[2] / "data" / "splice" / "splice.data"

# A, C, G and T, then the codes some sequences have for unsure bases. each code
# gets its own hv like any other base
BASES = "ACGTDNRS"
# the boundary where an exon ends and an intron starts, the reverse, or neither
CLASSES = ("EI", "IE", "N")
# the 58 windows of 3 bases in a row, as positions in the sequence
WINDOWS = np.arange(58)[:, None] + np.arange(3)


def load() -> tuple[np.ndarray, np.ndarray]:
    """all 3190 sequences of 60 bases and their classes. bases are numbered by
    their place in `BASES`, and classes by their place in `CLASSES`."""
    if not PATH.exists():
        _download()
    rows = [line.split(",") for line in PATH.read_text().splitlines()]
    sequences = np.array(
        [[BASES.index(b) for b in row[2].strip()] for row in rows]
    )
    labels = np.array([CLASSES.index(row[0]) for row in rows])
    return sequences, labels


def split(count: int) -> tuple[np.ndarray, np.ndarray]:
    """train and test indices, 80/20. it's the split in NysHD's code (Zhao et
    al., AAAI 2025), so we test on the same sequences they did."""
    # matches scikit-learn's `train_test_split(test_size=0.2, random_state=42)`
    # so `evolve.py` doesn't need scikit-learn
    shuffled = np.random.RandomState(42).permutation(count)
    test_count = math.ceil(0.2 * count)
    return np.sort(shuffled[test_count:]), np.sort(shuffled[:test_count])


def _download() -> None:
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_bytes(archive.read(PATH.name))
