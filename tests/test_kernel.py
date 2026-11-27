"""Tests for the kernel, which tells how alike a program makes each pair of
samples."""

import numpy as np

from pushhdc.kernel import Kernel
from tests.helpers import random_hvs, same_thing

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
