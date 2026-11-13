"""Evolves an encoder for splice junctions and reports its test accuracy.

Run with `uv run python experiments/splice/evolve.py`.
"""

import os

import numpy as np
from dataset import BASES, load, split

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

    # each position is a unit. the unit operations let evolution choose how the
    # positions make groups, for example the n-gram length, the gaps, or the
    # part of the sequence to use. there are 2 inputs. one is the hv of the
    # base at each position, and one is the hv of the position. each input has
    # the shape (sequences, 60 positions, dimension)
    rng = np.random.default_rng(SEED)
    hv = np.array([-1, 1], dtype=np.int8)
    values = rng.choice(hv, size=(len(BASES), DIMENSION))
    positions = rng.choice(hv, size=(sequences.shape[1], DIMENSION))
    bases = values[sequences]
    inputs = [bases, np.broadcast_to(positions, bases.shape)]

    evaluator = Evaluator(inputs, labels, train, seed=SEED, unit_ops=True)
    # in a sweep with 2 seeds, a tolerance of 0.01 gave the same
    # cross-validated accuracy as 0.005, and the default of 0.02 gave about
    # 0.03 less. on one seed, 0.01 also gave a smaller program and a shorter
    # run
    result = evolve(
        evaluator.spawner(),
        evaluator,
        seed=SEED,
        accuracy_tolerance=0.01,
        workers=os.cpu_count() or 1,
    )

    hvs, cost = evaluator.encode(result.best_genome)
    assert hvs is not None
    print(genome_to_code(result.best_genome).pretty_str())
    print(f"cost {cost}")
    print(f"cross-validated accuracy {result.best_fitness.accuracy:.3f}")
    print(f"test accuracy {centroid_accuracy(hvs, labels, train, test):.3f}")


if __name__ == "__main__":
    main()
