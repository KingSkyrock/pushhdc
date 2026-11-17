"""Evolves a multi-channel sequence encoder for promoter gene sequences.

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
K = 8  # Sliding 3-mer window size (e.g., codons/motifs)


def main() -> None:
    sequences, is_promoter = load()
    train, validation, test = split(is_promoter, seed=SEED)
    seen = np.concatenate([train, validation])

    rng = np.random.default_rng(SEED)
    
    # 1. Initialize random hypervectors for the 4 nucleotides: shape (4, dimension)
    values = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(NUM_NUCLEOTIDES, DIMENSION))
    
    # Map sequence characters to their hypervectors: shape (samples, 57, dimension)
    seq_hvs = values[sequences]
    num_samples, seq_len, dim = seq_hvs.shape

    # 2. Multi-channel input preparation:
    # Instead of 1 flat stream, we create K input streams for a sliding window of size K.
    # For seq_len = 57 and K = 3, we get 55 sliding units. 
    # inputs[0] = bases at offset 0
    # inputs[1] = bases at offset 1
    # inputs[2] = bases at offset 2
    num_units = seq_len - K + 1
    inputs = [
        seq_hvs[:, offset : num_units + offset]
        for offset in range(K)
    ]
    # Now `inputs` is a list of 3 arrays, each shaped (samples, 55, dimension).
    # PushGP can access them as input_0, input_1, and input_2 and bind them together!

    evaluator = Evaluator(inputs, is_promoter, seen, seed=SEED)
    result = evolve(
        evaluator.spawner(), evaluator, seed=SEED, workers=os.cpu_count() or 1
    )

    hvs, cost = evaluator.encode(result.best_genome)
    assert hvs is not None
    
    print("Evolved Push Program:")
    print(genome_to_code(result.best_genome).pretty_str())
    print(f"cost {cost}")
    print(f"cross-validated accuracy {result.best_fitness.accuracy:.3f}")
    print(f"test accuracy {centroid_accuracy(hvs, is_promoter, seen, test):.3f}")


if __name__ == "__main__":
    main()