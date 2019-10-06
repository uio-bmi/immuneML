import numpy as np


def jaccard(vector1, vector2):
    return np.sum(np.logical_and(vector1, vector2)) / np.sum(np.logical_or(vector1, vector2))
