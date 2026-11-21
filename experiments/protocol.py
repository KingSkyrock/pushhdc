"""The steps that every experiment shares, so their results stay comparable.

Evolution scores programs with one set of random hvs, the codebook. A
codebook makes each program look a little better or worse than it is, and
evolution favors the programs that look better. So the finalists get scored
again on new codebooks, and the chosen program gets reported on other
codebooks that played no part in the choice. Only that report uses the test
samples.
"""

import math
import os
from collections.abc import Callable, Sequence

import numpy as np

from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.evolution import choose, evolve
from pushhdc.inputs import Lookup
from pushhdc.pyshgp.gp.genome import genome_to_code

CHOICE_CODEBOOKS = range(200, 208)
REPORT_CODEBOOKS = range(100, 105)


def nyshd_split(count: int) -> tuple[np.ndarray, np.ndarray]:
    """train and test indices. this is the split in NysHD's code (Zhao et
    al., AAAI 2025), so the test samples are the same as theirs."""
    # matches scikit-learn's `train_test_split(test_size=0.2, random_state=42)`
    # so `evolve.py` does not need scikit-learn
    shuffled = np.random.RandomState(42).permutation(count)
    test_count = math.ceil(0.2 * count)
    return np.sort(shuffled[test_count:]), np.sort(shuffled[:test_count])


def evolve_and_report(
    inputs: Callable[[int], Sequence[np.ndarray | Lookup]],
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    seed: int,
    unit_ops: bool = False,
    lengths: Sequence[int] | None = None,
) -> None:
    """evolves an encoder on codebook `seed`, chooses one of its finalists on
    new codebooks, and prints it with its cross-validated and test accuracy
    on the report codebooks. `inputs` makes the inputs of all samples from
    the hvs of one codebook. each `Evaluator` also gets `unit_ops` and
    `lengths`."""

    # the number of a codebook also seeds its folds, because a program can
    # score high on one set of folds by chance, the same as on one codebook
    def evaluator(codebook: int) -> Evaluator:
        return Evaluator(
            inputs(codebook),
            labels,
            train,
            seed=codebook,
            unit_ops=unit_ops,
            lengths=lengths,
        )

    evolution = evaluator(seed)
    result = evolve(
        evolution.spawner(),
        evolution,
        seed=seed,
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
