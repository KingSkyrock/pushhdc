"""Tests for running the search with parallelism enabled.

Each test guards against a specific bug in parallel runs.
See docstrings for more info about tests.
Evaluators and instruction functions are defined at module level
because worker processes have to unpickle them.

What pickling is:

"pickling" is the process of converting a Python object hierarchy (like lists, 
dictionaries, or custom class instances) into a byte stream. This is used to
send objects to different parallel workers.
"Unpickling" is the inverse operation, where a byte stream is 
converted back into a functioning Python object hierarchy.
"""

import multiprocessing
import time

import numpy as np
import pytest

from pushhdc.pyshgp.gp.estimators import PushEstimator
from pushhdc.pyshgp.gp.evaluation import DatasetEvaluator, Evaluator
from pushhdc.pyshgp.gp.genome import GeneSpawner, Genome
from pushhdc.pyshgp.gp.individual import Individual
from pushhdc.pyshgp.gp.population import Population
from pushhdc.pyshgp.gp.search import (
    GeneticAlgorithm,
    ParallelContext,
    SearchConfiguration,
)
from pushhdc.pyshgp.push.atoms import Literal
from pushhdc.pyshgp.push.instruction_set import InstructionSet
from pushhdc.pyshgp.push.interpreter import PushInterpreter
from pushhdc.pyshgp.push.program import ProgramSignature
from pushhdc.pyshgp.push.types import PushInt
from pushhdc.pyshgp.validation import PushError

# Helpers


# counts how many times it gets pickled, i.e. sent to another process
class CountingEvaluator(DatasetEvaluator):
    pickles = 0

    def __getstate__(self):
        type(self).pickles += 1
        return self.__dict__.copy()


# every program gets the same error. program `0` takes 0.5 s
class AllTiedEvaluator(Evaluator):
    def evaluate(self, program):
        if program.code[0].value == 0:
            time.sleep(0.5)
        return np.array([1.0])


# small search for y = x^2 + 1 using integer instructions
def make_config(parallelism, evaluator_cls=DatasetEvaluator, seed=None):
    X =np.arange(-10, 10).reshape(-1, 1)
    y = [[int(x[0] ** 2 + 1)] for x in X]
    spawner = GeneSpawner(
        n_inputs=1,
        instruction_set=InstructionSet().register_core_by_stack({"int"}),
        literals=[],
        erc_generators=[],
    )
    return SearchConfiguration(
        signature=ProgramSignature(arity=1, output_stacks=["int"]),
        evaluator=evaluator_cls(X, y),
        spawner=spawner,
        population_size=20,
        max_generations=3,
        initial_genome_size=(5, 20),
        simplification_steps=5,
        parallelism=parallelism,
        seed=seed,
    )


# seeded search, returns its initial population and best program
def run_search(parallelism):
    config = make_config(parallelism, seed=0)
    try:
        ga = GeneticAlgorithm(config)
        initial = [indiv.program.pretty_str() for indiv in ga.population]
        best = ga.run()
    finally:
        config.tear_down()
    return initial, best.program.pretty_str(), list(best.error_vector)


# evaluates six tied programs and returns their order in the population
def evaluate_tied_population(parallelism):
    signature =ProgramSignature(arity=0, output_stacks=["int"])
    population = Population([
        Individual(Genome([Literal(value=i, push_type=PushInt)]), signature)
        for i in range(6)
    ])
    if parallelism:
        context = ParallelContext(AllTiedEvaluator(), parallelism)
        try:
            population.p_evaluate(context.pool)
        finally:
            context.close()
    else:
        population.evaluate(AllTiedEvaluator())
    return [indiv.program.code[0].value for indiv in population]


# parallel estimator whose programs read a second input that the data doesn't
# have, so fit raises
def make_exploding_estimator():
    instruction_set =InstructionSet(register_core=False)
    return PushEstimator(
        spawner=GeneSpawner(
            n_inputs=2,
            instruction_set=instruction_set,
            literals=[],
            erc_generators=[],
        ),
        interpreter=PushInterpreter(instruction_set),
        population_size=10,
        max_generations=2,
        parallelism=2,
    )


# Tests


def test_parallel_search_matches_serial():
    """a parallel search gives the same result as a serial one for one seed.

    compares the initial population, the best program and its errors. fails if
    the initial population is generated in worker processes, since they don't
    share the main process's random state. under fork every worker draws the
    same genomes, and under spawn/forkserver they ignore the seed.
    """
    serial = run_search(parallelism=False)
    parallel = run_search(parallelism=2)
    assert parallel == serial


def test_parallel_evaluation_keeps_order_of_tied_individuals():
    """parallel evaluation adds individuals in the order they were submitted.

    every program ties and program ``0`` is slow, so it finishes last when run
    in parallel. tied individuals are ordered by insertion, so taking results
    as they complete (``imap_unordered``) instead of in order makes the
    population order differ from a serial run, and selection diverges.
    """
    serial = evaluate_tied_population(parallelism=False)
    parallel = evaluate_tied_population(parallelism=2)
    assert parallel == serial


def test_evaluator_sent_at_most_once_per_worker():
    """the evaluator, which holds the dataset, reaches each worker only once.

    fails if the evaluator is pickled with every task. that re-sends the
    dataset for every individual, and made parallel runs on large datasets
    slower than serial ones.
    """
    CountingEvaluator.pickles = 0
    config = make_config(parallelism=2, evaluator_cls=CountingEvaluator)
    try:
        GeneticAlgorithm(config).run()
    finally:
        config.tear_down()
    assert CountingEvaluator.pickles <= 2


def test_fit_that_crashes_still_stops_worker_processes():
    """a fit that raises still shuts down its worker processes.

    the search raises because programs read an input that doesn't exist.
    fails if ``fit`` skips tear down when the search raises, or if tear down
    closes the pool without waiting for the workers to exit.
    """
    estimator = make_exploding_estimator()
    before = set(multiprocessing.active_children())
    with pytest.raises(PushError, match="IndexError"):
        estimator.fit(np.arange(10).reshape(-1, 1), np.arange(10).reshape(-1, 1))
    assert set(multiprocessing.active_children()) <= before
