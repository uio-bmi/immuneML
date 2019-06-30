from enum import Enum


class NormalizationType(Enum):

    RELATIVE_FREQUENCY = "l1"
    L2 = "l2"
    MAX = "max"
    BINARY = "binary"
    NONE = "none"
