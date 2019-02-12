import os
import pickle
from unittest import TestCase
from scipy import sparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC

from source.ml_methods.RandomForestClassifier import RandomForestClassifier


class TestRandomForestClassifier(TestCase):

    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        rfc = RandomForestClassifier()
        rfc.fit(sparse.csr_matrix(x), y)

    def test_predict(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        rfc = RandomForestClassifier()
        rfc.fit(sparse.csr_matrix(x), y)

        test_x = np.array([[0, 1, 0], [1, 0, 0]])
        y = rfc.predict(sparse.csr_matrix(test_x))["default"]

        self.assertTrue(len(y) == 2)
        self.assertTrue(y[0] in [0, 1, 2])
        self.assertTrue(y[1] in [0, 1, 2])

    def test_fit_by_cross_validation(self):
        x = sparse.csr_matrix(
            np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]]))
        y = np.array([[1, 0, 2, 0, 1, 0, 2, 0], [1, 0, 2, 0, 1, 0, 2, 0]])

        rfc = RandomForestClassifier()
        rfc.fit_by_cross_validation(x, y, number_of_splits=2, label_names=["t1", "t2"])

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        rfc = RandomForestClassifier()
        rfc.fit(sparse.csr_matrix(x), y)

        rfc.store("./")
        self.assertTrue(os.path.isfile("./random_forest_classifier.pkl"))

        with open("./random_forest_classifier.pkl", "rb") as file:
            rfc2 = pickle.load(file)

        self.assertTrue(isinstance(rfc2["default"], RFC))

        os.remove("./random_forest_classifier.pkl")
        os.remove("./rfc_optimal_params.json")

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        rfc = RandomForestClassifier()
        rfc.fit(sparse.csr_matrix(x), y)

        with open("./random_forest_classifier.pkl", "wb") as file:
            pickle.dump(rfc.get_model(), file)

        rfc2 = RandomForestClassifier()
        rfc2.load("./")

        self.assertTrue(isinstance(rfc2.get_model()["default"], RFC))

        os.remove("./random_forest_classifier.pkl")

