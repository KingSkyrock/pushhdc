"""Writes the baselines and an evolved encoder to `results/splice.txt`.

Run with `uv run --with scikit-learn python -m experiments.splice.report`.
"""

from experiments.protocol import write_report
from experiments.splice import baselines, evolve


def main() -> None:
    write_report("splice", baselines.main, evolve.main, evolve.DIMENSION)


if __name__ == "__main__":
    main()
