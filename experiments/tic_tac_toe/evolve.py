"""Evolves an encoder for tic-tac-toe boards and reports its test accuracy.

Run with `uv run python experiments/tic_tac_toe/evolve.py`.
"""

import sys
from pathlib import Path

import numpy as np
from dataset import LINES, load, split

# every experiment shares `protocol.py`, which is in `experiments/`
sys.path.append(str(Path(__file__).parents[1]))
from protocol import evolve_and_report

DIMENSION = 256
SEED = 0


def main() -> None:
    boards, x_won = load()
    train, validation, test = split(x_won, seed=SEED)
    # cross-validation does the validation split's job, so evolution sees
    # both. test stays out until the end
    seen = np.concatenate([train, validation])

    # we give evolution the 8 lines as units, so a program only has to encode
    # one line, and the same program gets reused on all 8. that makes the
    # search much easier, but evolution can't find a better structure than the
    # one we picked. later we want it to evolve the units too. the 3 inputs
    # are the hvs of each line's first, second and third square, each shaped
    # (boards, 8 lines, dimension)
    def inputs(codebook: int) -> list[np.ndarray]:
        rng = np.random.default_rng(codebook)
        values = rng.choice([-1.0, 1.0], size=(3, DIMENSION))  # blank, x, o
        squares = values[boards[:, LINES]]
        return [squares[:, :, i] for i in range(3)]

    evolve_and_report(inputs, x_won, seen, test, seed=SEED)


if __name__ == "__main__":
    main()
