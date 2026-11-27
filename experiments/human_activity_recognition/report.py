"""Writes the baselines table to `results/human_activity_recognition.txt`.

Run with
`uv run --with torch-hd --with scikit-learn python -m experiments.human_activity_recognition.report`.
"""

from experiments.human_activity_recognition import baselines
from experiments.protocol import write_report


def main() -> None:
    write_report("human_activity_recognition", baselines.main)


if __name__ == "__main__":
    main()
