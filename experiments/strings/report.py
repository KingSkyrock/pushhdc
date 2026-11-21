"""Writes the baselines and an evolved encoder to `results/<dataset>.txt`.

Run with `uv run --with scikit-learn python experiments/strings/report.py
protein`, or `sms` in place of `protein`.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import baselines
import evolve


def main() -> None:
    name = sys.argv[1]
    results = Path(__file__).parents[2] / "results" / f"{name}.txt"
    with redirect_stdout(io.StringIO()) as table:
        baselines.main()
        print(f"\nevolved encoder, dimension {evolve.DIMENSION}")
        evolve.main(name)
    results.parent.mkdir(exist_ok=True)
    results.write_text(table.getvalue())
    print(f"wrote {results}")


if __name__ == "__main__":
    main()
