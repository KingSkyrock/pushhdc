"""Writes the baselines and an evolved encoder to `results/splice.txt`.

Run with `uv run --with scikit-learn python experiments/splice/report.py`.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import baselines
import evolve

RESULTS = Path(__file__).parents[2] / "results" / "splice.txt"


def main() -> None:
    with redirect_stdout(io.StringIO()) as table:
        baselines.main()
        print(f"\nevolved encoder, dimension {evolve.DIMENSION}")
        evolve.main()
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(table.getvalue())
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
