"""Tests for seeding the search with `seed`.

The searches here use every selector, variation operator and gene type that
draws random numbers, so each of those draws runs.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pushhdc.pyshgp.gp.evaluation import DatasetEvaluator
from pushhdc.pyshgp.gp.genome import GeneSpawner, GeneTypes
from pushhdc.pyshgp.gp.search import (
    GeneticAlgorithm,
    SearchConfiguration,
    SimulatedAnnealing,
)
from pushhdc.pyshgp.gp.selection import (
    FitnessProportionate,
    Lexicase,
    Tournament,
)
from pushhdc.pyshgp.gp.variation import (
    Alternation,
    Genesis,
    get_variation_operator,
)
from pushhdc.pyshgp.push.instruction_set import InstructionSet
from pushhdc.pyshgp.push.program import ProgramSignature
from pushhdc.pyshgp.utils import DiscreteProbDistrib

REPO_ROOT = Path(__file__).resolve().parents[3]

# Helpers


def random_int(rng):
    return int(rng.integers(10))


def random_float(rng):
    return float(rng.random())


# a small search where every kind of random draw in pyshgp actually happens
# and can change the result, and where the instruction order would change if
# it depended on PYTHONHASHSEED. an erc (ephemeral random constant) is a
# random value that gets baked into a genome when its gene is made, and
# `erc_generators` are the functions that make them
def make_config(seed, selection, variation):
    X = np.column_stack([np.arange(-10, 10), np.arange(10, -10, -1)])
    y = [[int(x0**2 + x1)] for x0, x1 in X]
    distribution = DiscreteProbDistrib()
    for gene_type in GeneTypes:
        distribution.add(gene_type, 1)
    spawner = GeneSpawner(
        n_inputs=2,
        instruction_set=InstructionSet(register_core=True),
        literals=[1, 2],
        erc_generators=[random_int, random_float],
        distribution=distribution,
    )
    return SearchConfiguration(
        signature=ProgramSignature(arity=2, output_stacks=["int"]),
        evaluator=DatasetEvaluator(X, y),
        spawner=spawner,
        selection=selection,
        variation=variation,
        population_size=20,
        max_generations=5,
        initial_genome_size=(5, 20),
        simplification_steps=20,
        parallelism=False,
        seed=seed,
    )


# a summary of a seeded run, detailed enough that changing any random
# draw changes it. covers a genetic algorithm that mixes every selector and
# variation operator, plus simulated annealing
def summarize(seed):
    # makes lexicase's final random pick actually happen. the huge epsilon
    # keeps several candidates alive until then
    selection = (
        DiscreteProbDistrib()
        .add(Lexicase(), 1)
        .add(Lexicase(epsilon=1e9), 1)
        .add(Tournament(), 1)
        .add(FitnessProportionate(), 1)
    )
    variation = (
        DiscreteProbDistrib()
        .add(get_variation_operator("umad"), 1)
        .add(Alternation(), 1)
        .add(Genesis(size=(5, 20)), 1)
    )
    searches = [
        GeneticAlgorithm(make_config(seed, selection, variation)),
        SimulatedAnnealing(make_config(seed, "lexicase", "umad")),
    ]
    results = []
    for search in searches:
        best = search.run()
        population = [indiv.program.pretty_str() for indiv in search.population]
        results.append((population, best.program.pretty_str()))
    return repr(results)


# a fresh process starts numpy's global random state from the OS and takes
# its own PYTHONHASHSEED, so a run that leaks either one comes out different.
# returns summarize(0) and summarize(1)
def summaries_from_new_process(hash_seed):
    code = (
        "from tests.pyshgp.gp.test_seeding import summarize\n"
        "print(summarize(0))\n"
        "print(summarize(1))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


@pytest.fixture(scope="module")
def summaries():
    return {
        hash_seed: summaries_from_new_process(hash_seed)
        for hash_seed in (1, 2)
    }


# Tests


def test_same_seed_gives_same_search_in_new_processes(summaries):
    """two fresh processes with the same seed run identical searches.

    each process gets a different PYTHONHASHSEED, and numpy's global random
    state starts from the OS in each one. fails if any random draw skips the
    seeded generator, or if anything depends on hash ordering, like the
    registration order of the vector instructions.
    """
    assert summaries[1] == summaries[2]


def test_different_seeds_give_different_searches(summaries):
    """the seed controls the search.

    fails if the generator ignores `seed`, for example by always using the
    same fixed seed.
    """
    seed_0, seed_1 = summaries[1]
    assert seed_0 != seed_1


def test_search_does_not_touch_numpys_global_random_state():
    """a seeded search leaves `np.random`'s global state alone.

    fails if the search draws from or reseeds the global state, which would
    change the random numbers of any other code in the same process.
    """
    np.random.seed(123)
    expected = np.random.random()

    np.random.seed(123)
    summarize(0)

    assert np.random.random() == expected
