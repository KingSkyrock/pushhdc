"""Scores many genomes at a time, in worker processes when asked."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from multiprocessing import Pool
from typing import Any

from pushhdc.fitness import Evaluate, Fitness

_worker: dict[str, Evaluate] = {}


# keeps a dataset inside `evaluate` from being sent with every genome. it runs
# once as each worker process starts and hands it `evaluate`
def _init_worker(evaluate: Evaluate) -> None:
    _worker["evaluate"] = evaluate


def _evaluate_in_worker(genome: Any) -> Fitness:
    return _worker["evaluate"](genome)


@contextmanager
def scoring(
    evaluate: Evaluate, workers: int
) -> Generator[Callable[[list[Any]], list[Fitness]]]:
    """gives back a function that scores a list of genomes, in worker
    processes when `workers > 1`. results come back in order, which keeps
    parallel runs identical to serial ones. Python has to be able to copy
    `evaluate` to the worker processes, which works for a function or
    object defined at the top level of a module."""
    if workers <= 1:
        yield lambda genomes: [evaluate(genome) for genome in genomes]
        return
    pool = Pool(workers, initializer=_init_worker, initargs=(evaluate,))
    try:
        yield lambda genomes: list(pool.imap(_evaluate_in_worker, genomes))
    finally:
        pool.close()
        pool.join()
