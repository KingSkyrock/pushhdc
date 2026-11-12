"""Writes the baselines table to `results/human_activity_recognition.txt`.

Run with
`uv run --with torch-hd --with scikit-learn python experiments/human_activity_recognition/report.py`.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import baselines

RESULTS = Path(__file__).parents[2] / "results" / "human_activity_recognition.txt"


def main() -> None:
    with redirect_stdout(io.StringIO()) as table:
        baselines.main()
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(table.getvalue())
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
