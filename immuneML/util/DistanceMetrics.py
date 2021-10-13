import numpy as np


def jaccard(vector1, vector2, tmp_vector=None):
    return 1 - np.sum(np.logical_and(vector1, vector2, out=tmp_vector)) / np.sum(np.logical_or(vector1, vector2, out=tmp_vector))

def morisita_horn(vector1, vector2, *args, **kwargs):
    sum1 = np.sum(vector1)
    sum2 = np.sum(vector2)

    return 1 - ((2 * np.sum(vector1 * vector2)) / ((np.sum(vector1 ** 2) / (sum1 ** 2) + np.sum(vector2 ** 2) / (sum2**2)) * sum1 * sum2))
