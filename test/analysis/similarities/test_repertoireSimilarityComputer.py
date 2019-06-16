from unittest import TestCase

from scipy import sparse
import numpy as np

from source.analysis.similarities.RepertoireSimilarityComputer import RepertoireSimilarityComputer

class TestRepertoireSimilarityComputer(TestCase):


    def test_compute_pearson(self):
        a = sparse.csr_matrix(np.array([[0, 1, 1, 3, 4, 1, 5],
                                        [1, 1, 1, 0, 0, 5, 6]]))

        result = RepertoireSimilarityComputer.compute_pearson(a)

        self.assertTrue(np.equal(result[1, 0], 0.2554579258950239))
        self.assertTrue(np.equal(result[0, 1], 0.2554579258950239))
        self.assertTrue(np.equal(result[0, 0], 1))
        self.assertTrue(np.equal(result[1, 1], 1))

    def test_compute_morisita(self):
        a = sparse.csr_matrix(np.array([[0, 1, 1, 3, 4, 1, 5],
                                        [1, 1, 1, 0, 0, 5, 6]]))

        result = RepertoireSimilarityComputer.compute_morisita(a)

        self.assertTrue(np.equal(result[1, 0], 0.6269162497982895))
        self.assertTrue(np.equal(result[0, 1], 0.6269162497982895))
        self.assertTrue(np.equal(result[0, 0], 0.9999999999999999))
        self.assertTrue(np.equal(result[1, 1], 1.0000000000000002))

    def test_compute_cosine(self):
        a = sparse.csr_matrix(np.array([[0, 1, 1, 3, 4, 1, 5],
                                        [1, 1, 1, 0, 0, 5, 6]]))

        result = RepertoireSimilarityComputer.compute_cosine(a)

        self.assertTrue(np.equal(result[1, 0], 0.6352926082626867))
        self.assertTrue(np.equal(result[0, 1], 0.6352926082626867))
        self.assertTrue(np.equal(result[0, 0], 1.0000000000000002))
        self.assertTrue(np.equal(result[1, 1], 1.0))

    def test_compute_jaccard(self):
        a = sparse.csr_matrix(np.array([[0, 1, 1, 3, 4, 1, 5],
                                        [1, 1, 1, 0, 0, 5, 6]]))

        result = RepertoireSimilarityComputer.compute_jaccard(a)

        self.assertTrue(np.equal(result[1, 0], 0.5714285714285714))
        self.assertTrue(np.equal(result[0, 1], 0.5714285714285714))
        self.assertTrue(np.equal(result[0, 0], 1.0))
        self.assertTrue(np.equal(result[1, 1], 1.0))
