"""Evolves an encoder for tic-tac-toe boards and reports its test accuracy.

Run with `uv run python experiments/tic_tac_toe/evolve.py`.
"""

import os

import numpy as np
from dataset import LINES, load, split

from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.evolution import evolve
from pushhdc.pyshgp.gp.genome import genome_to_code

DIMENSION = 256
SEED = 0


def main() -> None:
    boards, x_won = load()
    train, validation, test = split(x_won, seed=SEED)
    # cross-validation does the validation split's job, so evolution sees
    # both. test stays out until the end
    seen = np.concatenate([train, validation])

    # we give evolution the 8 lines as units, so a program only has to encode
    # one line, and the same program gets reused on all 8. that makes the
    # search much easier, but evolution can't find a better structure than the
    # one we picked. later we want it to evolve the units too. the 3 inputs
    # are the hvs of each line's first, second and third square, each shaped
    # (boards, 8 lines, dimension)
    rng = np.random.default_rng(SEED)
    values = rng.choice([-1.0, 1.0], size=(3, DIMENSION))  # blank, x, o
    squares = values[boards[:, LINES]]
    inputs = [squares[:, :, i] for i in range(3)]

    evaluator = Evaluator(inputs, x_won, seen, seed=SEED)
    result = evolve(
        evaluator.spawner(), evaluator, seed=SEED, workers=os.cpu_count() or 1
    )

    hvs, cost = evaluator.encode(result.best_genome)
    assert hvs is not None
    print(genome_to_code(result.best_genome).pretty_str())
    print(f"cost {cost}")
    print(f"cross-validated accuracy {result.best_fitness.accuracy:.3f}")
    print(f"test accuracy {centroid_accuracy(hvs, x_won, seen, test):.3f}")


if __name__ == "__main__":
    main()
