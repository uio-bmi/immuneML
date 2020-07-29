from enum import Enum


class SplitType(Enum):

    K_FOLD = 0
    LOOCV = 1
    RANDOM = 2
    MANUAL = 3
    LEAVE_ONE_OUT_STRATIFICATION = 4
