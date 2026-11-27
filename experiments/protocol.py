"""The steps that every experiment shares, so their results stay comparable.

Evolution scores programs with one set of random hvs, the codebook. A
codebook makes each program look a little better or worse than it is, and
evolution favors the programs that look better. So the finalists get scored
again on new codebooks. With unit ops, the ints of the chosen program then get
tried at other values on the same codebooks (see `pushhdc.choice`). The
program gets reported on other codebooks that played no part in the choice.
Only that report uses the test samples.
"""

import io
import math
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from pushhdc.choice import choose, tune_ints
from pushhdc.encoder_ops import program_text
from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.evolution import evolve
from pushhdc.inputs import Lookup

CHOICE_CODEBOOKS = range(200, 208)
REPORT_CODEBOOKS = range(100, 105)

# `write_report` writes here, at the top of the repo
RESULTS = Path(__file__).parents[1] / "results"


def nyshd_split(count: int) -> tuple[np.ndarray, np.ndarray]:
    """train and test indices. this is the split in NysHD's code (Zhao et
    al., AAAI 2025), so the test samples are the same as theirs."""
    # matches scikit-learn's `train_test_split(test_size=0.2, random_state=42)`
    # so `evolve.py` does not need scikit-learn
    shuffled = np.random.RandomState(42).permutation(count)
    test_count = math.ceil(0.2 * count)
    return np.sort(shuffled[test_count:]), np.sort(shuffled[:test_count])


class Accuracy(NamedTuple):
    """the mean accuracy of an encoder over the report codebooks, in
    cross-validation on train and on test."""

    cross_validated: float
    test: float


@dataclass(frozen=True)
class Report:
    """the encoder that `evolve_and_report` gives back, with its cost, the
    number of finalists it was chosen from and its accuracy. with unit ops,
    `before_tuning` is the accuracy of the chosen encoder before its ints got
    tuned. it prints as the lines that the experiments print."""

    program: str
    cost: float
    finalists: int
    accuracy: Accuracy
    before_tuning: Accuracy | None

    def __str__(self) -> str:
        lines = [
            self.program,
            f"cost {self.cost}, chosen from {self.finalists} finalists",
            f"cross-validated accuracy {self.accuracy.cross_validated:.3f}",
            f"test accuracy {self.accuracy.test:.3f}",
        ]
        if self.before_tuning is not None:
            lines.append(
                f"before tuning ints: cross-validated accuracy "
                f"{self.before_tuning.cross_validated:.3f}, "
                f"test accuracy {self.before_tuning.test:.3f}"
            )
        return "\n".join(lines)


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
    workers: int | None = None,
) -> Report:
    """runs the shared protocol on one dataset and gives back what it found.
    it evolves an encoder on codebook `seed`, chooses one of the finalists on
    the choice codebooks and, with `unit_ops`, tunes the ints of that
    finalist on the same codebooks. the report holds the encoder with its
    cross-validated and test accuracy averaged over the report codebooks,
    and with `unit_ops` also the accuracy from before the tuning.

    `inputs(codebook)` makes the inputs of every sample from the hvs of that
    codebook, and each `Evaluator` also gets `unit_ops`, `lengths` and
    `sources`. scoring runs in `workers` processes, or in one for each CPU.
    """

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

    workers = workers or os.cpu_count() or 1
    search_evaluator = evaluator(seed)
    # tuning tries the same range of ints that new genes get in evolution
    top = search_evaluator.largest_int
    result = evolve(
        search_evaluator.spawner(), search_evaluator, seed=seed, workers=workers
    )
    # the evaluator of the search can be big, and it is not used again
    del search_evaluator
    chosen, fitness = choose(
        result.finalists, choice_evaluators, workers=workers
    )
    # without unit ops the spawner draws no ints, so there is no range of
    # ints to try
    genome = chosen
    if unit_ops:
        genome, fitness = tune_ints(
            chosen, choice_evaluators, top=top, workers=workers
        )

    # tuning can fix ints that evolution got wrong, and so hide a change in
    # how well evolution searches. so the accuracy from before the tuning
    # gets reported too
    reported = _report_accuracies(
        [genome, chosen] if unit_ops else [genome],
        evaluator,
        labels,
        train,
        test,
    )
    return Report(
        program=program_text(genome),
        cost=fitness.cost,
        finalists=len(result.finalists),
        accuracy=reported[0],
        before_tuning=reported[1] if unit_ops else None,
    )


def write_report(
    name: str,
    baselines: Callable[[], None],
    evolved: Callable[[], None] | None = None,
    dimension: int | None = None,
) -> None:
    """runs `baselines` and then `evolved`, and writes what they print to
    `results/<name>.txt`. `dimension` is the dimension of the evolved encoder,
    for the line above its report."""
    with redirect_stdout(io.StringIO()) as text:
        baselines()
        if evolved is not None:
            print(f"\nevolved encoder, dimension {dimension}")
            evolved()
    path = RESULTS / f"{name}.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text.getvalue())
    print(f"wrote {path}")


# the mean accuracy of each genome over the report codebooks. each report
# evaluator scores every genome before the next one gets made
def _report_accuracies(
    genomes: list[Any],
    evaluator: Callable[[int], Evaluator],
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> list[Accuracy]:
    cross_validated = np.zeros(len(genomes))
    tested = np.zeros(len(genomes))
    for codebook in REPORT_CODEBOOKS:
        report_evaluator = evaluator(codebook)
        for i, genome in enumerate(genomes):
            cross_validated[i] += report_evaluator(genome).accuracy
            hvs, _ = report_evaluator.encode(genome)
            assert hvs is not None
            tested[i] += centroid_accuracy(hvs, labels, train, test)
    cross_validated /= len(REPORT_CODEBOOKS)
    tested /= len(REPORT_CODEBOOKS)
    return [
        Accuracy(float(mean_cv), float(mean_test))
        for mean_cv, mean_test in zip(cross_validated, tested, strict=True)
    ]
