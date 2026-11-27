"""The raw sensor signals of UCI HAR, and their hvs.

Each window is 128 time steps with a reading from each of 9 sensor channels.
`dataset.py` loads the 561 numbers that people made from these signals. The
evolved encoders and `signal_baselines.py` read the raw signals instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.human_activity_recognition.dataset import ROOT, load
from pushhdc.levels import level_hvs, levels

CHANNELS = tuple(
    f"{sensor}_{axis}"
    for sensor in ("body_acc", "body_gyro", "total_acc")
    for axis in "xyz"
)
# each reading becomes one of this many levels
LEVELS = 100


def load_signals() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """the signals of all windows, shaped (windows, steps, channels), the
    label (0 to 5) and the person of each window, and the train and test
    windows. this is the official split, where train and test have different
    people."""
    # `load` also gets the data the first time
    _, train_labels, _, test_labels = load()
    signals = np.concatenate([_signals("train"), _signals("test")])
    people = np.concatenate([_people("train"), _people("test")])
    labels = np.concatenate([train_labels, test_labels])
    train = np.arange(len(train_labels))
    test = np.arange(len(train_labels), len(labels))
    return signals, labels, people, train, test


def readings(signals: np.ndarray, train: np.ndarray) -> list[np.ndarray]:
    """the level of each reading, one array shaped (windows, steps) for each
    channel. each channel gets its own levels, cut from its train readings
    only."""
    return [
        levels(channel, channel[train], LEVELS)
        for channel in np.moveaxis(signals, -1, 0)
    ]


def channel_hvs(
    codebook: int, dimension: int, steps: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """the level hvs of each channel and the hv of each time step, all made
    from codebook `codebook`. each channel gets its own level hvs, so the
    channels are not alike."""
    rng = np.random.default_rng(codebook)
    each_channel = [level_hvs(LEVELS, dimension, rng) for _ in CHANNELS]
    hv = np.array([-1, 1], dtype=np.int8)
    return each_channel, rng.choice(hv, size=(steps, dimension))


def _signals(split: str) -> np.ndarray:
    folder = ROOT / split / "Inertial Signals"
    return np.stack(
        [_table(folder / f"{channel}_{split}.txt") for channel in CHANNELS],
        axis=-1,
    )


def _people(split: str) -> np.ndarray:
    return _table(ROOT / split / f"subject_{split}.txt").ravel().astype(int)


def _table(path: Path) -> np.ndarray:
    return pd.read_csv(
        path, sep=r"\s+", header=None, dtype=np.float32, engine="c"
    ).to_numpy()
