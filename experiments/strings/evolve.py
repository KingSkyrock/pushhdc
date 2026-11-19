"""Evolves an encoder for one of NysHD's datasets of strings with different
lengths, and reports its test accuracy.

Run with `uv run python experiments/strings/evolve.py protein`, or `sms` in
place of `protein`.
"""

import os
import sys

import numpy as np
from datasets import LETTERS, load, padded, split

from pushhdc.evaluation import Evaluator, Lookup, centroid_accuracy
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


def main(name: str) -> None:
    strings, labels = load(name)
    # cross-validation does a validation split's job, so evolution sees all of
    # train. test stays out until the end
    train, test = split(len(labels))
    letters = padded(strings)
    lengths = [len(string) for string in strings]

    # each place in a string is a unit, as in the splice experiment. one input
    # is the hv of the letter at each place, as a `Lookup` of the letter hvs
    # and the letter numbers, so the padded array of hvs never gets made.
    # `lengths` tells the evaluator where each string ends, so it skips the
    # padding. the other input is the hv of the place, which is the same for
    # every string, so it is shaped (1, places, dimension). the folds change
    # with the codebook too, because a program can also score high on one
    # set of folds by chance
    def evaluator(codebook: int) -> Evaluator:
        rng = np.random.default_rng(codebook)
        hv = np.array([-1, 1], dtype=np.int8)
        values = np.concatenate(
            [
                rng.choice(hv, size=(len(LETTERS), DIMENSION)),
                np.zeros((1, DIMENSION), dtype=np.int8),
            ]
        )
        places = rng.choice(hv, size=(letters.shape[1], DIMENSION))
        inputs = [Lookup(values, letters), places[None]]
        return Evaluator(
            inputs,
            labels,
            train,
            seed=codebook,
            unit_ops=True,
            lengths=lengths,
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
        tested.append(centroid_accuracy(hvs, labels, train, test))
    print(genome_to_code(genome).pretty_str())
    print(f"cost {fitness.cost}, chosen from {len(result.finalists)} finalists")
    print(f"cross-validated accuracy {np.mean(cross_validated):.3f}")
    print(f"test accuracy {np.mean(tested):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
