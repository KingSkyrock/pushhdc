"""Baselines for UCI HAR, to compare evolved encoders against.

Run with
`uv run --with torch-hd --with scikit-learn python experiments/human_activity_recognition/baselines.py`.

First run downloads the UCI zip into `data/har/` (already done if that folder
exists). TorchHD id-level at high dimensions is slow on CPU; expect tens of
minutes for the full dimension x seed sweep.
"""

import sys

import torch
import torch.nn as nn
import torchhd
from dataset import N_CLASSES, N_FEATURES, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torchhd import embeddings
from torchhd.models import Centroid

DIMENSIONS = (1024, 4096, 10000)
SEEDS = range(5)
NUM_LEVELS = 100
# id-level materializes (batch, 561, dim); keep batches small to avoid thrashing
SINUSOID_BATCH = 256
ID_LEVEL_BATCH = 8


class SinusoidEncoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.project = embeddings.Sinusoid(N_FEATURES, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torchhd.hard_quantize(self.project(x))


class IdLevelEncoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.id = embeddings.Random(N_FEATURES, dimension)
        self.value = embeddings.Level(
            NUM_LEVELS, dimension, low=-1.0, high=1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_hv = torchhd.bind(self.id.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)


ENCODERS = (
    ("sinusoid", SinusoidEncoder, SINUSOID_BATCH),
    ("id_level", IdLevelEncoder, ID_LEVEL_BATCH),
)


def centroid_accuracy(
    encode: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
) -> float:
    with torch.no_grad():
        dimension = encode(x_train[:1]).shape[-1]
        model = Centroid(dimension, N_CLASSES)
        for start in range(0, len(x_train), batch_size):
            end = start + batch_size
            model.add(encode(x_train[start:end]), y_train[start:end])
        model.normalize()
        correct = 0
        for start in range(0, len(x_test), batch_size):
            end = start + batch_size
            outputs = model(encode(x_test[start:end]), dot=True)
            correct += int((outputs.argmax(1) == y_test[start:end]).sum())
    return correct / len(y_test)


def main() -> None:
    print("loading data...", file=sys.stderr, flush=True)
    x_train_np, y_train_np, x_test_np, y_test_np = load()
    x_train = torch.from_numpy(x_train_np.copy())
    y_train = torch.from_numpy(y_train_np.copy())
    x_test = torch.from_numpy(x_test_np.copy())
    y_test = torch.from_numpy(y_test_np.copy())

    def row(name: str, accuracies: list[float]) -> None:
        print(f"{name:32}" + "".join(f"{a:8.3f}" for a in accuracies))

    print("test accuracy")
    # laying / standing / walking are common; majority is the bar to beat
    majority = max(
        (y_test_np == label).mean() for label in range(N_CLASSES)
    )
    row("majority class", [float(majority)])
    models = {
        "logistic regression": LogisticRegression(max_iter=2000),
        "random forest": RandomForestClassifier(random_state=0, n_jobs=-1),
    }
    for name, model in models.items():
        print(f"fitting {name}...", file=sys.stderr, flush=True)
        model.fit(x_train_np, y_train_np)
        row(name, [float(model.score(x_test_np, y_test_np))])

    print(f"\n{'torchhd, by dimension':32}" + "".join(f"{d:8}" for d in DIMENSIONS))
    for name, encoder_cls, batch_size in ENCODERS:
        accuracies = []
        for dimension in DIMENSIONS:
            print(
                f"torchhd {name} d={dimension}...",
                file=sys.stderr,
                flush=True,
            )
            runs = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                encode = encoder_cls(dimension)
                encode.eval()
                runs.append(
                    centroid_accuracy(
                        encode,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        batch_size,
                    )
                )
            accuracies.append(float(sum(runs) / len(runs)))
        row(name, accuracies)


if __name__ == "__main__":
    main()
