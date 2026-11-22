"""Microbenchmark HDC ops and assign median-relative integer costs.

Times HyperVector methods backing hdc_instructions()
Costs use: max(1, round(10 * median_i / min_j(median_j))).

Examples:
  uv run python scripts/benchmark_instruction_costs.py
  uv run python scripts/benchmark_instruction_costs.py -n 500 -s 42 -d 10000
"""

import argparse
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pushhdc.hypervector import HyperVector

WARMUP_TRIALS = 10
COST_SCALE = 10


# Representation of each operation test case
@dataclass(frozen=True)
class OpSpec:
    name: str
    run: Callable[..., object]
    make_args: Callable[[np.random.Generator, int], tuple[object, ...]]


# Makes 2 random hvs for bind, bundle, and cosine sim
def _two_hvs(
    rng: np.random.Generator, dim: int
) -> tuple[HyperVector, HyperVector]:
    return (
        HyperVector.random(dimension=dim, rng=rng),
        HyperVector.random(dimension=dim, rng=rng),
    )


# Makes 1 random hv for permute
def _permute_args(
    rng: np.random.Generator, dim: int
) -> tuple[HyperVector, int]:
    shift = int(rng.integers(1, dim))
    return (HyperVector.random(dimension=dim, rng=rng), shift)


# Table of tested hv operations, add more below as new hv operations are created
OPS: tuple[OpSpec, ...] = (
    OpSpec("hv_bind", lambda a, b: a.bind(b), _two_hvs),
    OpSpec("hv_bundle", lambda a, b: a.bundle(b), _two_hvs),
    OpSpec(
        "hv_permute",
        lambda hv, shift: hv.permute(shift=shift),
        _permute_args,
    ),
    OpSpec(
        "hv_cosine_sim",
        lambda a, b: a.cosine_similarity(b),
        _two_hvs,
    ),
)


# Sample and return np array of raw runtimes
def _time_op(
    op: OpSpec, rng: np.random.Generator, dim: int, n: int
) -> np.ndarray:
    """Return per-trial runtimes in milliseconds."""
    for _ in range(WARMUP_TRIALS):
        op.run(*op.make_args(rng, dim))

    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        args = op.make_args(rng, dim)
        start = time.perf_counter()
        op.run(*args)
        samples[i] = (time.perf_counter() - start) * 1e3
    return samples


# Use np array of raw runtimes to calculate respective weighted cost
def _weighted_costs(medians: dict[str, float]) -> dict[str, int]:
    min_median = min(medians.values())
    if min_median <= 0:
        return dict.fromkeys(medians, COST_SCALE)
    return {
        name: max(1, round(COST_SCALE * median / min_median))
        for name, median in medians.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HDC instruction runtimes and print median-relative costs."
        )
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=1000,
        metavar="N",
        help="trials per operation (default: 1000)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="RNG seed (default: random at runtime)",
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=10000,
        metavar="D",
        help="hypervector dimension (default: 10000)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n < 1:
        raise SystemExit("n must be >= 1")
    if args.dim < 2:
        raise SystemExit("dim must be >= 2")

    seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(seed)

    means: dict[str, float] = {}  # Used for diagnosis only
    medians: dict[str, float] = {}
    for op in OPS:
        samples = _time_op(op, rng, args.dim, args.n)
        means[op.name] = float(np.mean(samples))
        medians[op.name] = float(np.median(samples))

    costs = _weighted_costs(medians)

    print(f"seed={seed}  n={args.n}  dim={args.dim}")
    print()
    print(f"{'operation':<16} {'mean_ms':>12} {'median_ms':>12} {'cost':>8}")
    for op in OPS:
        print(
            f"{op.name:<16} {means[op.name]:12.6f} "
            f"{medians[op.name]:12.6f} {costs[op.name]:8d}"
        )
    print()


if __name__ == "__main__":
    main()
