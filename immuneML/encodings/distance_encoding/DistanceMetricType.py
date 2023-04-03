from enum import Enum


class DistanceMetricType(Enum):

    JACCARD = "jaccard"
    MORISITA_HORN = "morisita_horn"
    TM_SCORE = "tm_score"
