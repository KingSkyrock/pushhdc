"""Evolves an encoder for promoter gene sequences and reports its test accuracy.

Run with `uv run python experiments/promoter_genes/evolve.py`.
"""

import os

import numpy as np
from dataset import load, split

from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.evolution import evolve
from pushhdc.pyshgp.gp.genome import genome_to_code

DIMENSION = 256
SEED = 0
NUM_NUCLEOTIDES = 4  # a, c, g, t


def main() -> None:
    sequences, is_promoter = load()
    train, validation, test = split(is_promoter, seed=SEED)
    # Cross-validation uses train + validation. Test stays out until the final evaluation.
    seen = np.concatenate([train, validation])

    rng = np.random.default_rng(SEED)
    
    # 1. Initialize random hypervectors for the 4 nucleotides: shape (4, dimension)
    values = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(NUM_NUCLEOTIDES, DIMENSION))
    
    # Map sequence characters to their respective hypervectors: shape (samples, seq_len, dimension)
    seq_hvs = values[sequences]
    num_samples, seq_len, dim = seq_hvs.shape

    # 2. Evaluator expects `inputs` to be a Sequence of 3D arrays, 
    # where each array is shaped (samples, units, dimension).
    # Let's package the sequence positions into units. If we want each position 
    # or sliding chunk to be a unit, let's shape it as (samples, 1, dimension) per input, 
    # or create a single 3D array and wrap it in a list.
    
    # Option A: A single input array of shape (samples, seq_len, dimension) wrapped in a list:
    inputs = [seq_hvs]  # Shape of seq_hvs is (samples, 57, dimension) -> 3D!

    evaluator = Evaluator(inputs, is_promoter, seen, seed=SEED)
    result = evolve(
        evaluator.spawner(), evaluator, seed=SEED, workers=os.cpu_count() or 1
    )

    hvs, cost = evaluator.encode(result.best_genome)
    assert hvs is not None
    
    print(genome_to_code(result.best_genome).pretty_str())
    print(f"cost {cost}")
    print(f"cross-validated accuracy {result.best_fitness.accuracy:.3f}")
    print(f"test accuracy {centroid_accuracy(hvs, is_promoter, seen, test):.3f}")


if __name__ == "__main__":
    main()