"""Evolves an encoder for splice junctions and reports its test accuracy.

Run with `uv run python experiments/splice/evolve.py`.
"""

import sys
from pathlib import Path

import numpy as np
from dataset import BASES, load

# every experiment shares `protocol.py`, which is in `experiments/`
sys.path.append(str(Path(__file__).parents[1]))
from protocol import evolve_and_report, nyshd_split

DIMENSION = 256
SEED = 0


def main() -> None:
    sequences, labels = load()
    # cross-validation does a validation split's job, so evolution sees all of
    # train. test stays out until the end
    train, test = nyshd_split(len(labels))

    # each position is a unit. the unit operations let evolution choose how the
    # positions make groups, for example the n-gram length, the gaps, or the
    # part of the sequence to use. one input is the hv of the base at each
    # position, shaped (sequences, positions, dimension). the other is the hv
    # of the position, which is the same for every sequence, so it is shaped
    # (1, positions, dimension)
    def inputs(codebook: int) -> list[np.ndarray]:
        rng = np.random.default_rng(codebook)
        hv = np.array([-1, 1], dtype=np.int8)
        values = rng.choice(hv, size=(len(BASES), DIMENSION))
        positions = rng.choice(hv, size=(sequences.shape[1], DIMENSION))
        return [values[sequences], positions[None]]

    evolve_and_report(inputs, labels, train, test, seed=SEED, unit_ops=True)


if __name__ == "__main__":
    main()
