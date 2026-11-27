"""Writes the baselines and an evolved encoder to `results/promoter_genes.txt`.

Run with `uv run --with scikit-learn python -m experiments.promoter_genes.report`.
"""

from experiments.promoter_genes import baselines, evolve
from experiments.protocol import write_report


def main() -> None:
    write_report(
        "promoter_genes", baselines.main, evolve.main, evolve.DIMENSION
    )


if __name__ == "__main__":
    main()
