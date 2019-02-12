from enum import Enum


class MetricType(Enum):

    # TODO: add more metrics

    ACCURACY = 1
    BALANCED_ACCURACY = 2
    CONFUSION_MATRIX = 4
    F1_MICRO = 5
    F1_MACRO = 6
    F1_WEIGHTED = 7
