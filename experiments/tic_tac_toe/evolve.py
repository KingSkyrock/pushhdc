"""Evolves an encoder for tic-tac-toe boards and reports its test accuracy.

Run with `uv run python experiments/tic_tac_toe/evolve.py`.
"""

import os

import numpy as np
from dataset import LINES, load, split

from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.evolution import choose, evolve
from pushhdc.pyshgp.gp.genome import genome_to_code

DIMENSION = 256
SEED = 0
# evolution uses one set of random hvs, the codebook. it makes each program
# look a little better or worse than it is, and evolution favors the programs
# that look better. so the finalists get scored again on new codebooks, and
# the chosen program gets reported on other codebooks that played no part in
# the choice
CHOICE_CODEBOOKS = range(200, 208)
REPORT_CODEBOOKS = range(100, 105)


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
    # the folds change with the codebook too, because a program can also get
    # lucky with one set of folds
    def evaluator(codebook: int) -> Evaluator:
        rng = np.random.default_rng(codebook)
        values = rng.choice([-1.0, 1.0], size=(3, DIMENSION))  # blank, x, o
        squares = values[boards[:, LINES]]
        inputs = [squares[:, :, i] for i in range(3)]
        return Evaluator(inputs, x_won, seen, seed=codebook)

    evolution = evaluator(SEED)
    result = evolve(
        evolution.spawner(), evolution, seed=SEED, workers=os.cpu_count() or 1
    )
    genome, fitness = choose(
        result.finalists,
        (evaluator(codebook) for codebook in CHOICE_CODEBOOKS),
    )

    cross_validated, tested = [], []
    for codebook in REPORT_CODEBOOKS:
        report = evaluator(codebook)
        cross_validated.append(report(genome).accuracy)
        hvs, _ = report.encode(genome)
        assert hvs is not None
        tested.append(centroid_accuracy(hvs, x_won, seen, test))
    print(genome_to_code(genome).pretty_str())
    print(f"cost {fitness.cost}, chosen from {len(result.finalists)} finalists")
    print(f"cross-validated accuracy {np.mean(cross_validated):.3f}")
    print(f"test accuracy {np.mean(tested):.3f}")


if __name__ == "__main__":
    main()
