import logging

import numpy as np


class HyperVector:
    """
    Represents a high-dimensional vector.
    """

    def __init__(self, data: np.ndarray, dimension: int = 10000) -> None:
        """
        Initializes from an existing NumPy array,
        or defaults to random bipolar initialization (-1 or +1).
        """
        if data is not None:
            self.data: np.ndarray = data.astype(np.float32)
            self.dimension: int = self.data.shape[0]
        else:
            self.dimension = dimension
            # Initialize with random bipolar values: {-1, +1}
            self.data = np.choice([-1.0, 1.0], size=(dimension,)).astype(
                np.float32
            )

    @classmethod
    def random(cls, dimension: int = 10000) -> "HyperVector":
        """Generate a new orthogonal random bipolar hypervector."""
        rng = np.random.default_rng()
        choices = np.array([-1.0, 1.0], dtype=np.float32)
        raw_data = rng.choice(choices, size=(dimension,))
        return cls(data=raw_data, dimension=dimension)

    def bind(self, other: "HyperVector") -> "HyperVector":
        """
        Performs Binding (multiplication component-wise)
        to associate two concepts (e.g., Feature-Value binding).
        """
        if self.dimension != other.dimension:
            raise ValueError(
                "Hypervectors must have the same dimension to bind."
            )

        bound_data = self.data * other.data
        return HyperVector(data=bound_data, dimension=self.dimension)

    def bundle(self, other: "HyperVector") -> "HyperVector":
        """
        Performs Bundling (superposition/addition)
        to store sets or sequences, binarizing back to bipolar {-1, +1}.
        """
        if self.dimension != other.dimension:
            raise ValueError(
                "Hypervectors must have the same dimension to bundle."
            )

        summed_data = self.data + other.data
        # Thresholding back to bipolar space {-1, +1}
        bundled_data = np.where(summed_data >= 0.0, 1.0, -1.0)
        return HyperVector(data=bundled_data, dimension=self.dimension)

    def permute(self, shift: int = 1) -> "HyperVector":
        """Performs Permutation (cyclic shift) to encode sequence order."""
        permuted_data = np.roll(self.data, shift=shift)
        return HyperVector(data=permuted_data, dimension=self.dimension)

    def cosine_similarity(self, other: "HyperVector") -> float:
        """
        Calculates cosine similarity (normalized dot product)
        between two hypervectors for similarity retrieval.
        """
        dot_product = np.dot(self.data, other.data)
        norm_self = np.linalg.norm(self.data)
        norm_other = np.linalg.norm(other.data)

        if norm_self == 0 or norm_other == 0:
            return 0.0

        return float(dot_product / (norm_self * norm_other))

    def __repr__(self) -> str:
        return f"HyperVector(dim={self.dimension}, sample={self.data[:5]}...)"


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # Test execution
    dim = 10000
    hv1 = HyperVector.random(dimension=dim)
    hv2 = HyperVector.random(dimension=dim)

    # Test Binding
    bound_hv = hv1.bind(hv2)
    logger.info(
        "Bound similarity (should be near 0): %.4f",
        bound_hv.cosine_similarity(hv1),
    )

    # Test Bundling
    set_hv = hv1.bundle(hv2)
    logger.info("Bundle similarity: %.4f", set_hv.cosine_similarity(hv1))

    # Test Permutation
    perm_hv = hv1.permute(shift=1)
    logger.info(
        "Permuted similarity (should be near 0): %.4f",
        perm_hv.cosine_similarity(hv1),
    )
