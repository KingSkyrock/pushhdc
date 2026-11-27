"""Baselines for NysHD's datasets of strings with different lengths.

Run with `uv run --with scikit-learn python -m experiments.strings.baselines
protein`, or `sms` in place of `protein`. The NysHD row needs a clone of
NysHD's code, see `experiments/nyshd.py`.
"""

import sys
from collections.abc import Callable
from functools import partial

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from experiments import nyshd
from experiments.protocol import nyshd_split
from experiments.strings.datasets import (
    ALPHABETS,
    LONGEST_K,
    letter_hvs,
    load,
    padded,
)
from pushhdc.evaluation import centroid_accuracy

DIMENSIONS = (64, 256, 1024, 4096, 10000)
SEEDS = range(5)
# the encoders below make arrays with a number for each letter of each string
# in each dimension, so they encode a part of the strings at a time, with
# about this many numbers in each part
PART = 50_000_000


# counts each letter, wherever it is
def bag_of_letters(
    letters: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    return letter_hvs(dimension, rng)[letters].sum(axis=1, dtype=np.int64)


# binds each letter to its place in the string
def record(
    letters: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    values = letter_hvs(dimension, rng)
    places = rng.choice(
        np.array([-1, 1], dtype=np.int8), (letters.shape[1], dimension)
    )
    return (values[letters] * places).sum(axis=1, dtype=np.int64)


# counts each run of n letters, wherever it is. a letter is permuted by its
# place in the run, so the order inside the run counts
def bag_of_ngrams(
    n: int, letters: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    hvs = letter_hvs(dimension, rng)[letters]
    runs = hvs[:, : hvs.shape[1] - n + 1]
    for place in range(1, n):
        runs = runs * np.roll(
            hvs[:, place : hvs.shape[1] - n + 1 + place], place, axis=-1
        )
    return runs.sum(axis=1, dtype=np.int64)


ENCODERS = {
    "bag_of_letters": bag_of_letters,
    "record": record,
    "bag_of_2grams": partial(bag_of_ngrams, 2),
    "bag_of_3grams": partial(bag_of_ngrams, 3),
}


# encodes a part of the strings at a time. every part starts from the same
# seed, so they all draw the same hvs
def in_parts(
    encode: Callable[[np.ndarray, int, np.random.Generator], np.ndarray],
    letters: np.ndarray,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seed = rng.integers(2**32)
    size = max(1, PART // (letters.shape[1] * dimension))
    return np.concatenate(
        [
            encode(
                letters[start : start + size],
                dimension,
                np.random.default_rng(seed),
            )
            for start in range(0, len(letters), size)
        ]
    )


def main(name: str) -> None:
    strings, labels = load(name)
    # nothing here gets tuned, so a validation split is not needed
    train, test = nyshd_split(len(labels))

    def row(label: str, accuracies: list[float]) -> None:
        print(f"{label:32}" + "".join(f"{a:8.3f}" for a in accuracies))

    print(f"{name}, test accuracy")
    counts = np.bincount(labels[train])
    row("most common class", [float((labels[test] == counts.argmax()).mean())])
    # counts of each letter and each pair of letters
    counted = CountVectorizer(
        analyzer="char", ngram_range=(1, 2), lowercase=False
    )
    features = counted.fit(np.array(strings)[train]).transform(strings)
    models = {
        "logistic regression": LogisticRegression(max_iter=1000),
        "random forest": RandomForestClassifier(random_state=0),
    }
    for label, model in models.items():
        model.fit(features[train], labels[train])
        row(label, [float(model.score(features[test], labels[test]))])

    # each seed draws different hvs, so hdc rows average over seeds
    print(f"\n{'hdc, by dimension':32}" + "".join(f"{d:8}" for d in DIMENSIONS))
    letters = padded(strings)
    encoders = [
        (label, partial(in_parts, encode, letters))
        for label, encode in ENCODERS.items()
    ]
    # NysHD picks its landmarks and its kernel setting from train, so it needs
    # to know the split. their code's setting is (2, 1, False) for protein
    # and (1, 1, False) for sms
    kernels = nyshd.kernels(strings, ALPHABETS[name], nyshd.settings(LONGEST_K))
    encode_nyshd, setting = nyshd.with_chosen_setting(kernels, labels, train)
    encoders.append(("nyshd", encode_nyshd))
    for label, encode in encoders:
        accuracies = []
        for dimension in DIMENSIONS:
            runs = []
            for seed in SEEDS:
                hvs = encode(dimension, np.random.default_rng(seed))
                runs.append(centroid_accuracy(hvs, labels, train, test))
            accuracies.append(float(np.mean(runs)))
        row(label, accuracies)
    print("\nnyshd kernel setting (k, g, apart), picked on train")
    for dimension in DIMENSIONS:
        print(f"{dimension:8}  {setting(dimension)}")


if __name__ == "__main__":
    main(sys.argv[1])
