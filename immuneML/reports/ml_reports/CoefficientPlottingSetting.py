from enum import Enum


class CoefficientPlottingSetting(Enum):
    ALL = "all"
    NONZERO = "nonzero"
    CUTOFF = "cutoff"
    N_LARGEST = "n_largest"
