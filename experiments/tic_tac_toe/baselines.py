"""Baselines for tic-tac-toe, to compare evolved encoders against.

Run with `uv run --with scikit-learn python experiments/tic_tac_toe/baselines.py`.
"""

import numpy as np
from dataset import LINES, load, split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pushhdc.evaluation import centroid_accuracy

DIMENSIONS = (64, 256, 1024, 4096, 10000)
SEEDS = range(5)


# one random hypervector of -1s and 1s per symbol. int8 keeps 10000-dimensional
# arrays small, and sums widen to int64 on their own
def random_hvs(n: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, dimension))


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def bundle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def permute(hv: np.ndarray, times: int = 1) -> np.ndarray:
    return np.roll(hv, times, axis=-1)


# the hypervectors of each line's first, second and third square, each shaped
# (boards, 8 lines, dimension). summing over axis 1 bundles the 8 lines
def line_squares(
    boards: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    squares = values[boards[:, LINES]]
    return squares[:, :, 0], squares[:, :, 1], squares[:, :, 2]


# the usual hdc encoding. binds each square's position to its value and
# bundles the 9 results
def record(
    boards: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    values = random_hvs(3, dimension, rng)
    positions = random_hvs(9, dimension, rng)
    return bind(positions, values[boards]).sum(axis=1)


# the rest are hand-written stand-ins for evolved programs. each runs the same
# few ops on every line and bundles the results
def bundle_each_line(
    boards: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    a, b, c = line_squares(boards, random_hvs(3, dimension, rng))
    return bundle(bundle(a, b), c).sum(axis=1)


def bind_each_line(
    boards: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    a, b, c = line_squares(boards, random_hvs(3, dimension, rng))
    return bind(bind(a, b), c).sum(axis=1)


# plain bind cancels repeats (x times x is all ones), so x x x comes out the
# same as x and two blanks. permuting by place in the line stops that
def bind_each_line_in_order(
    boards: np.ndarray, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    a, b, c = line_squares(boards, random_hvs(3, dimension, rng))
    return bind(bind(a, permute(b)), permute(c, 2)).sum(axis=1)


ENCODERS = (record, bundle_each_line, bind_each_line, bind_each_line_in_order)


def main() -> None:
    boards, x_won = load()
    train, validation, test = split(x_won, seed=0)
    train = np.concatenate([train, validation])

    def row(name: str, accuracies: list[float]) -> None:
        print(f"{name:32}" + "".join(f"{a:8.3f}" for a in accuracies))

    print("test accuracy")
    # x wins most games, so always guessing it is the bar to beat
    row("always x won", [x_won[test].mean()])
    one_hot = np.eye(3)[boards].reshape(len(boards), -1)
    models = {
        "logistic regression": LogisticRegression(max_iter=1000),
        "random forest": RandomForestClassifier(random_state=0),
    }
    for name, model in models.items():
        model.fit(one_hot[train], x_won[train])
        row(name, [model.score(one_hot[test], x_won[test])])

    # each seed draws different hypervectors, so hdc rows average over seeds
    print(f"\n{'hdc, by dimension':32}" + "".join(f"{d:8}" for d in DIMENSIONS))
    for encode in ENCODERS:
        accuracies = []
        for dimension in DIMENSIONS:
            runs = []
            for seed in SEEDS:
                hvs = encode(boards, dimension, np.random.default_rng(seed))
                runs.append(centroid_accuracy(hvs, x_won, train, test))
            accuracies.append(float(np.mean(runs)))
        row(encode.__name__, accuracies)


if __name__ == "__main__":
    main()
