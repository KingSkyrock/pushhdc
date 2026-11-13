"""Evolves an encoder for splice junctions and reports its test accuracy.

Run with `uv run python experiments/splice/evolve.py`.
"""

import os

import numpy as np
from dataset import BASES, WINDOWS, load, split

from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.evolution import evolve
from pushhdc.pyshgp.gp.genome import genome_to_code

DIMENSION = 256
SEED = 0


def main() -> None:
    sequences, labels = load()
    # cross-validation does a validation split's job, so evolution sees all of
    # train. test stays out until the end
    train, test = split(len(labels))

    # we give evolution the 58 windows of 3 bases as units, so a program only
    # has to encode one window, and the same program gets reused on all 58.
    # like the lines in tic-tac-toe, that makes the search much easier, but
    # evolution can't find a better structure than the one we picked. later we
    # want it to evolve the units too. the first 3 inputs are the hvs of each
    # window's first, second and third base, and the 4th is an hv for where the
    # window starts, which evolution can use or ignore. each is shaped
    # (sequences, 58 windows, dimension).
    rng = np.random.default_rng(SEED)
    hv = np.array([-1, 1], dtype=np.int8)
    values = rng.choice(hv, size=(len(BASES), DIMENSION))
    starts = rng.choice(hv, size=(len(WINDOWS), DIMENSION))
    bases = values[sequences[:, WINDOWS]]
    inputs = [bases[:, :, i] for i in range(3)]
    inputs.append(np.broadcast_to(starts, inputs[0].shape))

    evaluator = Evaluator(inputs, labels, train, seed=SEED)
    result = evolve(
        evaluator.spawner(), evaluator, seed=SEED, workers=os.cpu_count() or 1
    )

    hvs, cost = evaluator.encode(result.best_genome)
    assert hvs is not None
    print(genome_to_code(result.best_genome).pretty_str())
    print(f"cost {cost}")
    print(f"cross-validated accuracy {result.best_fitness.accuracy:.3f}")
    print(f"test accuracy {centroid_accuracy(hvs, labels, train, test):.3f}")


if __name__ == "__main__":
    main()
