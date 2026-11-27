"""Writes the baselines and an evolved encoder to `results/tic_tac_toe.txt`.

Run with `uv run --with scikit-learn python -m experiments.tic_tac_toe.report`.
"""

from experiments.protocol import write_report
from experiments.tic_tac_toe import baselines, evolve


def main() -> None:
    write_report("tic_tac_toe", baselines.main, evolve.main, evolve.DIMENSION)


if __name__ == "__main__":
    main()
