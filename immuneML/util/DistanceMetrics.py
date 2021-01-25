import numpy as np


def jaccard(vector1, vector2, tmp_vector=None):
    return np.sum(np.logical_and(vector1, vector2, out=tmp_vector)) / np.sum(np.logical_or(vector1, vector2, out=tmp_vector))
