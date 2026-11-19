"""Tests for clearing, which keeps new ideas ahead of copies of better ones."""

import numpy as np

from pushhdc.clearing import Kernel, clear
from pushhdc.evolution import Fitness

# Helpers


# random hvs for as many samples as `Evaluator` uses as probes
def random_hvs(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice([-1, 1], size=(64, 256))


# the same hvs after a fixed bind and permute, plus a bias for every sample.
# a program with these does the same thing as one with `hvs`
def same_thing(hvs: np.ndarray) -> np.ndarray:
    key = np.random.default_rng(100).choice([-1, 1], size=hvs.shape[1])
    return np.roll(hvs * key, 3, axis=1) + 5


# Tests


def test_kernel_ignores_a_fixed_bind_permute_and_bias() -> None:
    """hvs that differ only by a fixed bind and permute and a bias for every
    sample have kernels that align at 1.

    fails if the kernel compares the hvs themselves in place of how alike they
    make each pair of samples, or if a bias for every sample counts.
    """
    hvs = random_hvs(0)
    kernel, same = Kernel.of(hvs), Kernel.of(same_thing(hvs))
    assert kernel is not None
    assert same is not None
    assert np.isclose(kernel.alignment(same), 1.0)


def test_unrelated_kernels_align_near_zero() -> None:
    """kernels of unrelated random hvs align near 0, also on as few samples
    as the probes.

    fails if the estimate keeps how alike each sample is with itself. with
    few samples that part is large, and unrelated programs then look alike.
    """
    a, b = Kernel.of(random_hvs(0)), Kernel.of(random_hvs(1))
    assert a is not None
    assert b is not None
    assert abs(a.alignment(b)) < 0.1


def test_hvs_that_tell_no_samples_apart_have_no_kernel() -> None:
    """hvs that are the same for every sample, or too few samples to
    estimate, give no kernel.

    fails if those give a kernel, which divides by zero when it is built or
    compared.
    """
    assert Kernel.of(np.full((64, 256), 3)) is None
    assert Kernel.of(random_hvs(0)[:3]) is None


def test_clearing_moves_copies_behind_new_ideas() -> None:
    """a genome that does the same thing as a better one goes behind the
    genomes that do something new, and each group keeps its order.

    fails if copies keep their place, if the best genome moves, if the order
    inside a group changes, or if each genome with no kernel counts as new.
    """
    hvs = random_hvs(0)
    genomes = ["best", "copy of best", "new idea", "blank", "another blank"]
    kernels = [
        Kernel.of(hvs),
        Kernel.of(same_thing(hvs)),
        Kernel.of(random_hvs(1)),
        None,
        None,
    ]
    ranked = [(genome, Fitness(0.9, 1)) for genome in genomes]
    order = [genome for genome, _ in clear(ranked, kernels, 0.8)]
    assert order == [
        "best",
        "new idea",
        "blank",
        "copy of best",
        "another blank",
    ]
