from enum import Enum


class OperationType(Enum):

    IN = 0
    NOT_IN = 1
    NOT_NA = 2
    GREATER_THAN = 3
    LESS_THAN = 4
    TOP_N = 5
    RANDOM_N = 6
