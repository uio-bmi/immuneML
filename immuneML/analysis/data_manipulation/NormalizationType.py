from enum import Enum


class NormalizationType(Enum):
    """
    Different normalization types for vectors.

    - RELATIVE_FREQUENCY: Each value is divided by the sum of all values (L1 normalization).
    - L2: Each value is divided by the L2 norm (Euclidean norm) [focuses on the direction of the vector, not the magnitude].
    - MAX: Each value is divided by the maximum value in the vector.
    - BINARY: Each value is set to 1 if it is greater than 0, otherwise it is set to 0.
    - NONE: No normalization is applied.

    Used in encodings like GeneFrequencyEncoder, KmerFrequencyEncoder, etc.
    """

    RELATIVE_FREQUENCY = "l1"
    L2 = "l2"
    MAX = "max"
    BINARY = "binary"
    NONE = "none"
