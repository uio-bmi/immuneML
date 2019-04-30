import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.LogisticRegression import LogisticRegression
from source.util.PathBuilder import PathBuilder


class TestLogisticRegression(TestCase):
    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        lr = LogisticRegression()
        lr.fit(sparse.csr_matrix(x), y, ["test"])

    def test_predict(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([[1, 0, 2, 0], [1, 0, 2, 0]])

        lr = LogisticRegression()
        lr.fit(sparse.csr_matrix(x), y, ["test1", "test2"])

        test_x = np.array([[0, 1, 0], [1, 0, 0]])
        y = lr.predict(sparse.csr_matrix(test_x), ["test1", "test2"])

        self.assertTrue(len(y["test1"]) == 2)
        self.assertTrue(y["test1"][0] in [0, 1, 2])
        self.assertTrue(y["test2"][1] in [0, 1, 2])

    def test_fit_by_cross_validation(self):
        x = sparse.csr_matrix(
            np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]]))
        y = np.array([[1, 0, 2, 0, 1, 0, 2, 0], [1, 0, 2, 0, 1, 0, 2, 0]])

        lr = LogisticRegression()
        lr.fit_by_cross_validation(x, y, number_of_splits=2, label_names=["test1", "test2"])

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        lr = LogisticRegression()
        lr.fit(sparse.csr_matrix(x), y)

        path = EnvironmentSettings.root_path + "test/tmp/lr/"

        lr.store(path)
        self.assertTrue(os.path.isfile(path + "logistic_regression.pickle"))

        with open(path + "logistic_regression.pickle", "rb") as file:
            lr2 = pickle.load(file)

        self.assertTrue(isinstance(lr2["default"], SGDClassifier))

        shutil.rmtree(path)

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = np.array([1, 0, 2, 0])

        lr = LogisticRegression()
        lr.fit(sparse.csr_matrix(x), y)

        path = EnvironmentSettings.root_path + "test/tmp/lr2/"
        PathBuilder.build(path)

        with open(path + "logistic_regression.pickle", "wb") as file:
            pickle.dump(lr.get_model(), file)

        lr2 = LogisticRegression()
        lr2.load(path)

        self.assertTrue(isinstance(lr2.get_model()["default"], SGDClassifier))

        shutil.rmtree(path)

