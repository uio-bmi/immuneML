from enum import Enum


class AssessmentType(Enum):

    k_fold = 0
    loocv = 1
    random = 2
    random_balanced = 3
