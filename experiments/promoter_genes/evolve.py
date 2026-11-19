"""Evolves an encoder for promoter gene sequences and reports its test accuracy.

Run with `uv run python experiments/promoter_genes/evolve.py`.
"""

import os

import numpy as np
from dataset import load, split

from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.evolution import choose, evolve
from pushhdc.pyshgp.gp.genome import genome_to_code

DIMENSION = 256
SEED = 0
NUM_NUCLEOTIDES = 4  # a, c, g, t
# evolution uses one set of random hvs, the codebook. it makes each program
# look a little better or worse than it is, and evolution favors the programs
# that look better. so the finalists get scored again on new codebooks, and
# the chosen program gets reported on other codebooks that played no part in
# the choice
CHOICE_CODEBOOKS = range(200, 208)
REPORT_CODEBOOKS = range(100, 105)


def main() -> None:
    sequences, is_promoter = load()
    # cross-validation does a validation split's job, so evolution sees all of
    # train. test stays out until the end
    train, test = split(len(is_promoter))

    # each position is a unit, as in the splice experiment. the unit
    # operations let evolution choose how the positions make groups, for
    # example the k-mer length, the gaps, or the part of the sequence to use,
    # so the inputs do not have to fix a window size. one input is the hv of
    # the nucleotide at each position, shaped (sequences, positions,
    # dimension). the other is the hv of the position, which is the same for
    # every sequence, so it is shaped (1, positions, dimension). the folds
    # change with the codebook too, because a program can also get lucky
    # with one set of folds
    def evaluator(codebook: int) -> Evaluator:
        rng = np.random.default_rng(codebook)
        hv = np.array([-1, 1], dtype=np.int8)
        values = rng.choice(hv, size=(NUM_NUCLEOTIDES, DIMENSION))
        positions = rng.choice(hv, size=(sequences.shape[1], DIMENSION))
        inputs = [values[sequences], positions[None]]
        return Evaluator(
            inputs, is_promoter, train, seed=codebook, unit_ops=True
        )

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
        tested.append(centroid_accuracy(hvs, is_promoter, train, test))

    print("Evolved Push Program:")
    print(genome_to_code(genome).pretty_str())
    print(f"cost {fitness.cost}, chosen from {len(result.finalists)} finalists")
    print(f"cross-validated accuracy {np.mean(cross_validated):.3f}")
    print(f"test accuracy {np.mean(tested):.3f}")


if __name__ == "__main__":
    main()
