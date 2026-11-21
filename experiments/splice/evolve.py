"""Evolves an encoder for splice junctions and reports its test accuracy.

Run with `uv run python experiments/splice/evolve.py`.
"""

import os

import numpy as np
from dataset import BASES, load, split

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
    sequences, labels = load()
    # cross-validation does a validation split's job, so evolution sees all of
    # train. test stays out until the end
    train, test = split(len(labels))

    # each position is a unit. the unit operations let evolution choose how the
    # positions make groups, for example the n-gram length, the gaps, or the
    # part of the sequence to use. one input is the hv of the base at each
    # position, shaped (sequences, positions, dimension). the other is the hv
    # of the position, which is the same for every sequence, so it is shaped
    # (1, positions, dimension). the folds change with the codebook too,
    # because a program can also get lucky with one set of folds
    def evaluator(codebook: int) -> Evaluator:
        rng = np.random.default_rng(codebook)
        hv = np.array([-1, 1], dtype=np.int8)
        values = rng.choice(hv, size=(len(BASES), DIMENSION))
        positions = rng.choice(hv, size=(sequences.shape[1], DIMENSION))
        inputs = [values[sequences], positions[None]]
        return Evaluator(inputs, labels, train, seed=codebook, unit_ops=True)

    evolution = evaluator(SEED)
    result = evolve(
        evolution.spawner(),
        evolution,
        seed=SEED,
        workers=os.cpu_count() or 1,
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
        tested.append(centroid_accuracy(hvs, labels, train, test))
    print(genome_to_code(genome).pretty_str())
    print(f"cost {fitness.cost}, chosen from {len(result.finalists)} finalists")
    print(f"cross-validated accuracy {np.mean(cross_validated):.3f}")
    print(f"test accuracy {np.mean(tested):.3f}")


if __name__ == "__main__":
    main()
