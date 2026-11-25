"""The steps that every experiment shares, so their results stay comparable.

Evolution scores programs with one set of random hvs, the codebook. A
codebook makes each program look a little better or worse than it is, and
evolution favors the programs that look better. So the finalists get scored
again on new codebooks. With unit ops, the ints of the chosen program then get
tried at other values on the same codebooks (see `pushhdc.tuning`). The
program gets reported on other codebooks that played no part in the choice.
Only that report uses the test samples.
"""

import math
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.evolution import choose, evolve
from pushhdc.inputs import Lookup
from pushhdc.pyshgp.gp.genome import genome_to_code
from pushhdc.tuning import tune_ints

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
    sources: Sequence[Any] | np.ndarray | None = None,
) -> None:
    """evolves an encoder on codebook `seed`, chooses one of its finalists on
    new codebooks, and with `unit_ops` tunes the ints of that finalist on the
    same codebooks. then it prints the encoder with its cross-validated and
    test accuracy on the report codebooks, and with `unit_ops` also the
    accuracy from before the tuning. `inputs` makes the inputs of all samples
    from the hvs of one codebook. each `Evaluator` also gets `unit_ops`,
    `lengths` and `sources`."""

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
            sources=sources,
        )

    # `choose` and `tune_ints` score on the same new codebooks
    def choice_evaluators() -> Iterator[Evaluator]:
        return (evaluator(codebook) for codebook in CHOICE_CODEBOOKS)

    workers = os.cpu_count() or 1
    evolution = evaluator(seed)
    # tuning tries the same range of ints that new genes get in evolution
    top = evolution.largest_int
    result = evolve(evolution.spawner(), evolution, seed=seed, workers=workers)
    # the evaluator of evolution can be big, and it is not used again
    del evolution
    chosen, fitness = choose(result.finalists, choice_evaluators())
    # without unit ops the spawner draws no ints, so there is no range of
    # ints to try
    genome = chosen
    if unit_ops:
        genome, fitness = tune_ints(
            chosen, choice_evaluators, top=top, workers=workers
        )

    # tuning can fix ints that evolution got wrong, and so hide a change in
    # how well evolution searches. so the accuracy from before the tuning
    # gets printed too
    reported = [genome, chosen] if unit_ops else [genome]
    cross_validated = np.zeros(len(reported))
    tested = np.zeros(len(reported))
    for codebook in REPORT_CODEBOOKS:
        report = evaluator(codebook)
        for i, each in enumerate(reported):
            cross_validated[i] += report(each).accuracy
            hvs, _ = report.encode(each)
            assert hvs is not None
            tested[i] += centroid_accuracy(hvs, labels, train, test)
    cross_validated /= len(REPORT_CODEBOOKS)
    tested /= len(REPORT_CODEBOOKS)
    print(genome_to_code(genome).pretty_str())
    print(f"cost {fitness.cost}, chosen from {len(result.finalists)} finalists")
    print(f"cross-validated accuracy {cross_validated[0]:.3f}")
    print(f"test accuracy {tested[0]:.3f}")
    if unit_ops:
        print(
            f"before tuning ints: cross-validated accuracy "
            f"{cross_validated[1]:.3f}, test accuracy {tested[1]:.3f}"
        )
