"""Evolves an encoder for one of NysHD's datasets of strings with different
lengths, and reports its test accuracy.

Run with `uv run python experiments/strings/evolve.py protein`, or `sms` in
place of `protein`.
"""

import sys
from pathlib import Path

import numpy as np
from datasets import letter_hvs, load, padded

from pushhdc.inputs import Lookup

# every experiment shares `protocol.py`, which is in `experiments/`
sys.path.append(str(Path(__file__).parents[1]))
from protocol import evolve_and_report, nyshd_split

DIMENSION = 256
SEED = 0


def main(name: str) -> None:
    strings, labels = load(name)
    # cross-validation does a validation split's job, so evolution sees all of
    # train. test stays out until the end
    train, test = nyshd_split(len(labels))
    letters = padded(strings)
    lengths = [len(string) for string in strings]

    # each place in a string is a unit, as in the splice experiment. one input
    # is the hv of the letter at each place, as a `Lookup` of the letter hvs
    # and the letter numbers, so the padded array of hvs never gets made.
    # `lengths` tells the evaluator where each string ends, so it skips the
    # padding. the other input is the hv of the place, which is the same for
    # every string, so it is shaped (1, places, dimension)
    def inputs(codebook: int) -> list[np.ndarray | Lookup]:
        rng = np.random.default_rng(codebook)
        values = letter_hvs(DIMENSION, rng)
        hv = np.array([-1, 1], dtype=np.int8)
        places = rng.choice(hv, size=(letters.shape[1], DIMENSION))
        return [Lookup(values, letters), places[None]]

    evolve_and_report(
        inputs, labels, train, test, seed=SEED, unit_ops=True, lengths=lengths
    )


if __name__ == "__main__":
    main(sys.argv[1])
