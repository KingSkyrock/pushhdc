"""Evolves an encoder for UCI HAR from the raw sensor signals, and reports its
test accuracy.

Run with `uv run python experiments/human_activity_recognition/evolve.py`.
"""

import sys
from pathlib import Path

import numpy as np
from signals import channel_hvs, load_signals, readings

from pushhdc.inputs import Lookup

# every experiment shares `protocol.py`, which is in `experiments/`
sys.path.append(str(Path(__file__).parents[1]))
from protocol import evolve_and_report

DIMENSION = 256
SEED = 0
# a worker can hold many hvs of all the train windows at the same time, so
# fewer workers than CPUs keep the memory in check
WORKERS = 3


def main() -> None:
    signals, labels, people, train, test = load_signals()
    levels = readings(signals, train)
    steps = signals.shape[1]

    # each time step of a window is a unit, as each place of a string is. each
    # channel is one input, a `Lookup` of its level hvs and the level of each
    # reading, so the hvs of all its units get made only while a program
    # reads them. the last input is the hv of each time step, which is the
    # same for every window, so it is shaped (1, steps, dimension)
    def inputs(codebook: int) -> list[np.ndarray | Lookup]:
        each_channel, step_hvs = channel_hvs(codebook, DIMENSION, steps)
        return [
            *(
                Lookup(hvs, level)
                for hvs, level in zip(each_channel, levels, strict=True)
            ),
            step_hvs[None],
        ]

    # the windows overlap by half, and each person makes many windows. so
    # each fold holds all the windows of its people, and a program gets no
    # credit for telling people apart
    evolve_and_report(
        inputs,
        labels,
        train,
        test,
        seed=SEED,
        unit_ops=True,
        sources=people,
        workers=WORKERS,
    )


if __name__ == "__main__":
    main()
