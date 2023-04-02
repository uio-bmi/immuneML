import numpy as np


def jaccard(vector1, vector2, tmp_vector=None):
    return 1 - np.sum(np.logical_and(vector1, vector2, out=tmp_vector)) / np.sum(np.logical_or(vector1, vector2, out=tmp_vector))


def morisita_horn(vector1, vector2, *args, **kwargs):
    sum1 = np.sum(vector1)
    sum2 = np.sum(vector2)

    return 1 - ((2 * np.sum(vector1 * vector2)) / ((np.sum(vector1 ** 2) / (sum1 ** 2) + np.sum(vector2 ** 2) / (sum2 ** 2)) * sum1 * sum2))


def levenshtein(vector1, vector2, *args, **kwargs):
    m, n = len(vector1), len(vector2)
    d = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if vector1[i - 1] == vector2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    return d[m][n]
