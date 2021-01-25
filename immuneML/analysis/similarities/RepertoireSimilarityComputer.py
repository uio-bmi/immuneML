import numpy as np
from scipy import sparse
from sklearn import preprocessing


class RepertoireSimilarityComputer:

    @staticmethod
    def compute_pearson(a):

        a = a.astype(np.float64)
        n = a.shape[1]

        # Compute the covariance matrix
        rowsum = a.sum(1)
        centering = rowsum.dot(rowsum.T.conjugate()) / n
        C = (a.dot(a.T.conjugate()) - centering) / (n - 1)

        # The correlation coefficients are given by
        # C_{i,j} / sqrt(C_{i} * C_{j})
        d = np.diag(C)
        coeffs = C / np.sqrt(np.outer(d, d))

        return coeffs

    @staticmethod
    def compute_morisita(a):
        # Works on unnormalized or relative frequency normalized values only - not e.g. L2 normalized
        xy = a * a.T
        repertoire_totals = a.sum(axis=1).A1
        pairwise_mult_repertoire_totals = repertoire_totals[:, None] * repertoire_totals[None, :]
        repertoire_frequency = sparse.diags(1 / a.sum(axis=1).A.ravel()) @ a
        repertoire_frequency.data **= 2
        simpson_diversity = repertoire_frequency.sum(axis=1).A1
        pairwise_sum_simpson = simpson_diversity[:, None] + simpson_diversity[None, :]
        return 2 * (xy / pairwise_sum_simpson) / pairwise_mult_repertoire_totals

    @staticmethod
    def compute_jaccard(a):

        a = a.T

        a.data[:] = 1
        cols_sum = a.getnnz(axis=0)
        ab = a.T * a

        # for rows
        aa = np.repeat(cols_sum, ab.getnnz(axis=0))
        # for columns
        bb = cols_sum[ab.indices]

        similarities = ab.copy()

        similarities.data = similarities.data / (aa + bb - ab.data)

        return similarities

    @staticmethod
    def compute_cosine(a):

        a = a.T

        col_normed_mat = preprocessing.normalize(a.tocsc(), axis=0)

        return (col_normed_mat.T * col_normed_mat).A
