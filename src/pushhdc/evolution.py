"""Evolutionary search over Push genomes, plus pyshgp and HDC smoke tests."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

import numpy as np

from pushhdc.hypervector import HyperVector
from pushhdc.push_hdc import hdc_instruction_set
from pushhdc.pyshgp.gp.genome import Genome
from pushhdc.pyshgp.gp.variation import Alternation, get_variation_operator
from pushhdc.pyshgp.push.atoms import CodeBlock, Literal
from pushhdc.pyshgp.push.config import PushConfig
from pushhdc.pyshgp.push.interpreter import PushInterpreter
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.pyshgp.push.type_library import PushTypeLibrary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fitness:
    """how good a genome is. higher accuracy is better, then lower cost."""

    accuracy: float
    cost: float


@dataclass(frozen=True)
class Result:
    """the best genome of a run, its fitness, and each generation's best."""

    best_genome: Any
    best_fitness: Fitness
    history: list[Fitness]


type Evaluate = Callable[[Any], Fitness]
type Ranked = list[tuple[Any, Fitness]]


def evolve(
    spawner: Any,
    evaluate: Evaluate,
    *,
    seed: int | None,
    population_size: int = 100,
    generations: int = 50,
    initial_genome_size: tuple[int, int] = (10, 50),
    max_genome_length: int = 100,
    elites: int = 2,
    tournament_size: int = 7,
    crossover_rate: float = 0.5,
    accuracy_tolerance: float = 0.0,
    workers: int = 1,
) -> Result:
    """evolve genomes toward higher accuracy, then lower cost.

    with `workers > 1`, `evaluate` runs in worker processes, so Python has to
    be able to copy it to them, which works for a function or object defined
    at the top level of a module. rounding accuracy to a multiple of
    `accuracy_tolerance` lets cost decide between genomes that are about
    equally accurate. `history` has the best fitness of each generation,
    starting with the random first one.
    """
    rng = np.random.default_rng(seed)
    umad = get_variation_operator("umad")
    alternation = Alternation()

    def make_child(ranked: Ranked) -> Any:
        parent = _tournament(ranked, tournament_size, rng)
        if rng.random() < crossover_rate:
            other = _tournament(ranked, tournament_size, rng)
            parent = alternation.produce([parent, other], spawner, rng=rng)
        child = umad.produce([parent], spawner, rng=rng)
        return Genome(child[:max_genome_length])

    population = [
        spawner.spawn_genome(initial_genome_size, rng=rng)
        for _ in range(population_size)
    ]
    with _evaluator(evaluate, workers) as evaluate_all:
        ranked = _rank(
            list(zip(population, evaluate_all(population), strict=True)),
            accuracy_tolerance,
        )
        history = [ranked[0][1]]
        for _ in range(generations):
            children = [
                make_child(ranked) for _ in range(population_size - elites)
            ]
            scored = list(zip(children, evaluate_all(children), strict=True))
            ranked = _rank(ranked[:elites] + scored, accuracy_tolerance)
            history.append(ranked[0][1])

    best_genome, best_fitness = ranked[0]
    return Result(best_genome, best_fitness, history)


# sorts genomes best first, by rounded accuracy and then cost
def _rank(scored: Ranked, accuracy_tolerance: float) -> Ranked:
    def key(pair: tuple[Any, Fitness]) -> tuple[float, float]:
        accuracy = pair[1].accuracy
        if accuracy_tolerance > 0:
            accuracy = round(accuracy / accuracy_tolerance)
        return (-accuracy, pair[1].cost)

    return sorted(scored, key=key)


# picks `size` random genomes and returns the best of them. `ranked` is sorted
# best first, which makes the lowest picked index the winner
def _tournament(ranked: Ranked, size: int, rng: np.random.Generator) -> Any:
    picks = rng.choice(len(ranked), size=min(size, len(ranked)), replace=False)
    return ranked[int(picks.min())][0]


_worker: dict[str, Evaluate] = {}


# keeps a dataset inside `evaluate` from being sent with every genome. it runs
# once as each worker process starts and hands it `evaluate`
def _init_worker(evaluate: Evaluate) -> None:
    _worker["evaluate"] = evaluate


def _evaluate_in_worker(genome: Any) -> Fitness:
    return _worker["evaluate"](genome)


# gives back a function that scores a list of genomes, in worker processes when
# `workers > 1`. results come back in order, which keeps parallel runs
# identical to serial ones
@contextmanager
def _evaluator(
    evaluate: Evaluate, workers: int
) -> Generator[Callable[[list[Any]], list[Fitness]]]:
    if workers <= 1:
        yield lambda genomes: [evaluate(genome) for genome in genomes]
        return
    pool = Pool(workers, initializer=_init_worker, initargs=(evaluate,))
    try:
        yield lambda genomes: list(pool.imap(_evaluate_in_worker, genomes))
    finally:
        pool.close()
        pool.join()


def run_pyshgp_smoke_test() -> list[int]:
    """Run a minimal Push program and return its integer output."""
    type_library = PushTypeLibrary()
    program = Program(
        code=CodeBlock([Literal(value=42, push_type=type_library["int"])]),
        signature=ProgramSignature(arity=0, output_stacks=["int"]),
    )
    output = PushInterpreter().run(program, inputs=[])
    if output != [42]:
        raise RuntimeError(f"Unexpected PyshGP output: {output!r}")
    return [int(value) for value in output]


def run_hdc_smoke_test(dimension: int = 256) -> float:
    """Bind two HVs on the Push stack and return cosine(sim of result, first).

    Exercises hypervector type registration, stack I/O, and ``hv_bind``.
    """
    rng = np.random.default_rng(0)
    instruction_set = hdc_instruction_set()
    type_library = instruction_set.type_library
    hv_type = type_library["hypervector"]

    hv_a = HyperVector.random(dimension=dimension, rng=rng)
    hv_b = HyperVector.random(dimension=dimension, rng=rng)

    program = Program(
        code=CodeBlock(
            [
                Literal(value=hv_a, push_type=hv_type),
                Literal(value=hv_b, push_type=hv_type),
                instruction_set["hv_bind"].meta(),
                Literal(value=hv_a, push_type=hv_type),
                instruction_set["hv_cosine_sim"].meta(),
            ]
        ),
        signature=ProgramSignature(
            arity=0,
            output_stacks=["float"],
            push_config=PushConfig(rng=rng),
        ),
    )
    output = PushInterpreter(instruction_set).run(program, inputs=[])
    if len(output) != 1 or not isinstance(output[0], float):
        raise RuntimeError(f"Unexpected HDC Push output: {output!r}")
    return float(output[0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pyshgp_smoke_test()
    sim = run_hdc_smoke_test()
    logger.info("hdc smoke cosine(bound, a)=%.4f", sim)
