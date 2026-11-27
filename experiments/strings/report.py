"""Writes the baselines and an evolved encoder to `results/<dataset>.txt`.

Run with `uv run --with scikit-learn python -m experiments.strings.report
protein`, or `sms` in place of `protein`.
"""

import sys

from experiments.protocol import write_report
from experiments.strings import baselines, evolve


def main() -> None:
    name = sys.argv[1]
    write_report(
        name,
        lambda: baselines.main(name),
        lambda: evolve.main(name),
        evolve.DIMENSION,
    )


if __name__ == "__main__":
    main()
