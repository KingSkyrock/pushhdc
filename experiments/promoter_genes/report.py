"""Writes the baselines table to `results/promoter_genes.txt`.

Run with `uv run --with scikit-learn python experiments/promoter_genes/report.py`.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import baselines

RESULTS = (
    Path(__file__).parents[2]
    / "experiments"
    / "promoter_genes"
    / "results"
    / "promoter_genes.txt"
)


def main() -> None:
    with redirect_stdout(io.StringIO()) as table:
        baselines.main()
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(table.getvalue())
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
