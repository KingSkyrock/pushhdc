"""Evolves an encoder for splice-junction gene sequences and reports its test accuracy.

Run with `uv run python experiments/splice_junctions/evolve.py`.
"""

import os

import numpy as np
from dataset import load, split

from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.evolution import evolve
from pushhdc.pyshgp.gp.genome import genome_to_code

DIMENSION = 256
SEED = 0
NUM_NUCLEOTIDES = 5  # a, c, g, t, n


def main() -> None:
    sequences, labels = load()
    train, validation, test = split(labels, seed=SEED)
    seen = np.concatenate([train, validation])

    rng = np.random.default_rng(SEED)
    
    # Initialize random hypervectors for nucleotides: shape (5, dimension)
    values = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(NUM_NUCLEOTIDES, DIMENSION))
    
    # Map sequence characters to their respective hypervectors: shape (samples, 60, dimension)
    seq_hvs = values[sequences]

    # Evaluator expects a sequence of 3D arrays shaped (samples, units, dimension)
    inputs = [seq_hvs]

    evaluator = Evaluator(inputs, labels, seen, seed=SEED)
    result = evolve(
        evaluator.spawner(), evaluator, seed=SEED, workers=os.cpu_count() or 1
    )

    hvs, cost = evaluator.encode(result.best_genome)
    assert hvs is not None
    
    print(genome_to_code(result.best_genome).pretty_str())
    print(f"cost {cost}")
    print(f"cross-validated accuracy {result.best_fitness.accuracy:.3f}")
    print(f"test accuracy {centroid_accuracy(hvs, labels, seen, test):.3f}")


if __name__ == "__main__":
    main()