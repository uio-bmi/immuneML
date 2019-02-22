import os
import pickle
from unittest import TestCase
from scipy import sparse
import numpy as np
from sklearn.linear_model import SGDClassifier

from source.ml_methods.SVM import SVM


class TestSVM(TestCase):
    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        svm = SVM()
        svm.fit(sparse.csr_matrix(x), y)

    def test_predict(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        svm = SVM()
        svm.fit(sparse.csr_matrix(x), y, ["test"])

        test_x = np.array([[0, 1, 0], [1, 0, 0]])
        y = svm.predict(sparse.csr_matrix(test_x))["test"]

        self.assertTrue(len(y) == 2)
        self.assertTrue(y[0] in [0, 1, 2])
        self.assertTrue(y[1] in [0, 1, 2])

    def test_fit_by_cross_validation(self):
        x = sparse.csr_matrix(np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]]))
        y = np.array([[1, 0, 2, 0, 1, 0, 2, 0], [1, 0, 2, 0, 1, 0, 2, 0]])

        svm = SVM()
        svm.fit_by_cross_validation(x, y, number_of_splits=2, label_names=["t1", "t2"])

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        svm = SVM()
        svm.fit(sparse.csr_matrix(x), y)

        svm.store("./")
        self.assertTrue(os.path.isfile("./svm.pickle"))

        with open("./svm.pickle", "rb") as file:
            svm2 = pickle.load(file)

        self.assertTrue(isinstance(svm2["default"], SGDClassifier))

        os.remove("./svm.pickle")
        os.remove("./svm.json")

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        svm = SVM()
        svm.fit(sparse.csr_matrix(x), y)

        with open("./svm.pickle", "wb") as file:
            pickle.dump(svm.get_model(), file)

        svm2 = SVM()
        svm2.load("./")

        self.assertTrue(isinstance(svm2.get_model()["default"], SGDClassifier))

        os.remove("./svm.pickle")
