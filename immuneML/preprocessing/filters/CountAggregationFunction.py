from enum import Enum


class CountAggregationFunction(Enum):
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    FIRST = "first"
    LAST = "last"
