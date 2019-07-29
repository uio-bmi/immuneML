from enum import Enum


class SplitType(Enum):

    K_FOLD = 0
    LOOCV = 1
    RANDOM = 2
    RANDOM_BALANCED = 3
