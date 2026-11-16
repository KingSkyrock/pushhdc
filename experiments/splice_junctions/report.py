"""Writes the baselines table to `experiments/splice_junctions/results/splice_junctions.txt`.

Run with `uv run --with scikit-learn python experiments/splice_junctions/report.py`.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import baselines

RESULTS = Path(__file__).parent / "results" / "splice_junctions.txt"


def main() -> None:
    with redirect_stdout(io.StringIO()) as table:
        baselines.main()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(table.getvalue())
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()