"""Loads the UCI HAR dataset (official train/test split)."""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ROOT = Path(__file__).parents[2] / "data" / "har"
TRAIN_X = ROOT / "train" / "X_train.txt"
TRAIN_Y = ROOT / "train" / "y_train.txt"
TEST_X = ROOT / "test" / "X_test.txt"
TEST_Y = ROOT / "test" / "y_test.txt"

N_FEATURES = 561
N_CLASSES = 6
ACTIVITIES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """official train/test features and labels. labels are 0..5 for the six
    activities. features are already scaled to about [-1, 1]."""
    if not TRAIN_X.exists():
        _download()
    x_train = _read_features(TRAIN_X)
    y_train = _read_labels(TRAIN_Y)
    x_test = _read_features(TEST_X)
    y_test = _read_labels(TEST_Y)
    return x_train, y_train, x_test, y_test


def _read_features(path: Path) -> np.ndarray:
    return pd.read_csv(
        path, sep=r"\s+", header=None, dtype=np.float32, engine="c"
    ).to_numpy()


def _read_labels(path: Path) -> np.ndarray:
    return (
        pd.read_csv(path, sep=r"\s+", header=None, dtype=np.int64, engine="c")
        .to_numpy()
        .ravel()
        - 1
    )


def _download() -> None:
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    ROOT.mkdir(parents=True, exist_ok=True)
    archive.extractall(ROOT)
    _flatten_extracted()


def _flatten_extracted() -> None:
    """UCI's outer zip may nest another zip or a `UCI HAR Dataset` folder.
    end up with `data/har/train` and `data/har/test`."""
    nested_zip = next(ROOT.rglob("*HAR*.zip"), None)
    if nested_zip is not None and nested_zip.is_file():
        with zipfile.ZipFile(nested_zip) as archive:
            archive.extractall(ROOT)
        nested_zip.unlink()

    if TRAIN_X.exists():
        return

    dataset_dir = next(
        (p for p in ROOT.rglob("X_train.txt") if p.is_file()),
        None,
    )
    if dataset_dir is None:
        raise FileNotFoundError(
            f"could not find X_train.txt under {ROOT} after download"
        )
    # X_train.txt lives in .../train/; move that train/test parent up to ROOT
    source_root = dataset_dir.parents[1]
    for name in ("train", "test"):
        src = source_root / name
        dest = ROOT / name
        if src.exists() and src.resolve() != dest.resolve():
            if dest.exists():
                _rmtree(dest)
            src.rename(dest)

    # drop leftover nested folders / readmes we don't need for loading
    for child in list(ROOT.iterdir()):
        if child.name not in ("train", "test") and child.is_dir():
            _rmtree(child)


def _rmtree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _rmtree(child)
        else:
            child.unlink()
    path.rmdir()
