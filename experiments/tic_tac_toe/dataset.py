"""Loads the UCI Tic-Tac-Toe Endgame dataset and splits it."""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://archive.ics.uci.edu/static/public/101/tic+tac+toe+endgame.zip"
PATH = Path(__file__).parents[2] / "data" / "tic-tac-toe" / "tic-tac-toe.data"

# the 8 ways to get three in a row. squares are numbered left to right, top to
# bottom
LINES = np.array([
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6],
])  # fmt: skip


def load() -> tuple[np.ndarray, np.ndarray]:
    """all 958 endgame boards and whether x won. squares are 0 for blank, 1 for
    x and 2 for o."""
    if not PATH.exists():
        _download()
    rows = [line.split(",") for line in PATH.read_text().split()]
    boards = np.array([["bxo".index(s) for s in row[:9]] for row in rows])
    x_won = np.array([row[9] == "positive" for row in rows])
    return boards, x_won


def split(
    x_won: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """train, validation and test indices, 60/20/20 in the same class mix."""
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for label in (True, False):
        indices = rng.permutation(np.flatnonzero(x_won == label))
        train_end = round(0.6 * len(indices))
        validation_end = round(0.8 * len(indices))
        train.extend(indices[:train_end])
        validation.extend(indices[train_end:validation_end])
        test.extend(indices[validation_end:])
    return np.sort(train), np.sort(validation), np.sort(test)


def _download() -> None:
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_bytes(archive.read(PATH.name))
