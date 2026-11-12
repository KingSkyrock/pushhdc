"""
PushHDC HyperVector Engine (Deferred Binarization & int8 Accumulation)
----------------------------------------------------------------------
This module implements a Vector Symbolic Architecture (VSA) hypervector
engine optimized for evolutionary Genetic Programming (PushGP).

Architectural Strategy:
1. Intermediate Efficiency: Vectors are stored as native NumPy int8 arrays
   (-128 to +127) rather than float32, reducing memory by 4x.
2. Deferred Binarization: Intermediate operations (like bundling) perform
   pure integer addition without heavy hashing, random number generation,
   or binarization at every step.
3. Final Boundary Resolution: Binarization and Seeded Tie-Breaking are
   deferred strictly to the output boundary (when evaluating fitness or
   storing vectors in memory).
"""

import logging

import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)


class HyperVector:
    """
    Represents a high-dimensional hypervector supporting integer-valued
    accumulation (int8) for fast intermediate PushGP operations and deferred binarization.
    """

    # Explicitly disable Python object hashing since NumPy arrays are mutable
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        data: np.ndarray | None = None,
        dimension: int = 10000,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Initializes a HyperVector instance.
        Args:
            data (np.ndarray | None): Existing array data. If provided, cast to int8.
            dimension (int): Dimensionality of the hypervector (default: 10,000).
            rng (np.random.Generator | None): Required if generating a random vector
                to ensure deterministic reproducibility.
        """
        if data is not None:
            # Cast input data to int8 for memory efficiency (1 byte per dimension)
            self.data: np.ndarray = data.astype(np.int8)
            self.dimension: int = self.data.shape[0]
        else:
            if rng is None:
                raise ValueError("rng is required to make a random hypervector")
            random_hv = self.random(dimension=dimension, rng=rng)
            self.data = random_hv.data
            self.dimension = random_hv.dimension

    @classmethod
    def random(
        cls, dimension: int = 10000, *, rng: np.random.Generator
    ) -> "HyperVector":
        """
        Generates a new orthogonal random bipolar hypervector.
        Lifecycle Phase: Base Initialization (Generation 0)
        Samples uniformly from {-1, 1} using a seeded random generator.
        """
        choices = np.array([-1, 1], dtype=np.int8)
        raw_data = rng.choice(choices, size=(dimension,))
        return cls(data=raw_data, dimension=dimension)

    def bind(self, other: "HyperVector") -> "HyperVector":
        """
        Performs Binding (component-wise multiplication).
        Purpose: Associates two concepts (e.g., Role * Value, Feature * Color).
        Math: Multiplies int8 elements (-1 * -1 = 1, 1 * -1 = -1).
        Results remain strictly within bipolar {-1, 1} bounds.
        """
        if self.dimension != other.dimension:
            raise ValueError(
                "Hypervectors must have the same dimension to bind."
            )

        bound_data = self.data * other.data
        return HyperVector(data=bound_data, dimension=self.dimension)

    def bundle(self, other: "HyperVector") -> "HyperVector":
        """
        Performs Deferred Bundling (element-wise integer addition).

        Lifecycle Phase: Intermediate PushGP Execution (Recursive Tree Operations)
        Purpose: Superimposes vectors to represent sets or collections.
        Performance Optimization: Avoids binarization, hashing, and PRNG tie-breaking
          entirely during intermediate steps, making this a single fast addition.

        OVERFLOW LIMIT:
        Signed int8 holds values between -128 and +127. If bundling pure {-1, 1} vectors,
        you can safely accumulate up to ~127 additions before hitting integer overflow.
        If both intermediate vectors hold non-binary sums, this limit can be reached sooner.
        Recommendation: Monitor the depth of recursive bundling.
        """
        if self.dimension != other.dimension:
            raise ValueError(
                "Hypervectors must have the same dimension to bundle."
            )

        # Element-wise integer addition (accumulating votes)
        summed_data = self.data + other.data
        return HyperVector(data=summed_data, dimension=self.dimension)

    def permute(self, shift: int = 1) -> "HyperVector":
        """
        Performs Permutation (cyclic roll).
        Purpose: Encodes sequence order, position, or syntax structure
        Math: Shifts array elements cyclically using np.roll.
        """
        permuted_data = np.roll(self.data, shift=shift)
        return HyperVector(data=permuted_data, dimension=self.dimension)

    def binarize(self, rng: np.random.Generator) -> "HyperVector":
        """
        Converts accumulated int8 sums back into clean bipolar {-1, 1} vectors.

        Lifecycle Phase: Output Boundary / Fitness Evaluation / Memory Storage
        When to Call: Executed before computing similarity or storing a reference vector.

        Thresholds positive sums to 1 and non-positive sums to -1.
        Seeded Tie-Breaking: Resolves exact zeros (ties where sum == 0)
        using a deterministic SHA-256 / PRNG seed to guarantee 100%
        reproducibility across PushGP generations.
        """
        # Step 1: Base mapping -> positive sums become 1, zero/negative sums become -1
        bundled_data = np.where(self.data > 0, 1, -1).astype(np.int8)

        # Step 2: Handle exact zeros (ties) where accumulated sum == 0
        zeros = self.data == 0
        if np.any(zeros):
            num_zeros = np.count_nonzero(zeros)
            # Apply seeded random choice to ensure deterministic tie-breaking
            bundled_data[zeros] = rng.choice(
                np.array([-1, 1], dtype=np.int8), size=num_zeros
            )

        return HyperVector(data=bundled_data, dimension=self.dimension)

    def cosine_similarity(self, other: "HyperVector") -> float:
        """
        Calculates normalized cosine similarity between two hypervectors.

        Lifecycle Phase: Fitness Evaluation / Retrieval
        - Purpose: Measures geometric orientation alignment (score between -1.0 and 1.0).
        - Implementation Note: Casts int8 data to float64 for dot product and norm
          calculations to prevent intermediate integer overflow.
        """
        d1 = self.data.astype(np.float64)
        d2 = other.data.astype(np.float64)

        dot_product = np.dot(d1, d2)
        norm_self = np.linalg.norm(d1)
        norm_other = np.linalg.norm(d2)

        if norm_self == 0 or norm_other == 0:
            return 0.0

        return float(dot_product / (norm_self * norm_other))

    def __eq__(self, other: object) -> bool:
        """Equality comparison checking dimension and exact underlying data arrays."""
        if not isinstance(other, HyperVector):
            return NotImplemented
        return self.dimension == other.dimension and np.array_equal(
            self.data, other.data
        )

    def __repr__(self) -> str:
        """Concise string representation displaying dimension and a sample of the data."""
        return f"HyperVector(dim={self.dimension}, sample={self.data[:5]}...)"


if __name__ == "__main__":
    # Test execution block demonstrating the deferred binarization pipeline
    dim = 10000
    rng = np.random.default_rng(0)

    hv1 = HyperVector.random(dimension=dim, rng=rng)
    hv2 = HyperVector.random(dimension=dim, rng=rng)
    hv3 = HyperVector.random(dimension=dim, rng=rng)

    # 1. Intermediate Operations (Fast int8 accumulation, no binarization)
    bound_hv = hv1.bind(hv2)
    accumulated_hv = bound_hv.bundle(hv3)  # Pure int8 addition

    # 2. Output Boundary: Binarize and tie-break only at the very end
    final_hv = accumulated_hv.binarize(rng)

    # 3. Test Retrieval / Similarity
    logger.info(
        "Final similarity to hv1: %.4f",
        final_hv.cosine_similarity(hv1),
    )
