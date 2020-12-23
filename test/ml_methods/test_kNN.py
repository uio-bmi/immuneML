import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier

from source.caching.CacheType import CacheType
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.KNN import KNN
from source.util.PathBuilder import PathBuilder


class TestKNN(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"test": np.array([1, 0, 2, 0])}

        knn = KNN()
        knn.fit(EncodedData(examples=sparse.csr_matrix(x), labels=y), "test")

    def test_predict(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"test1": [1, 0, 2, 0], "test2": [1, 0, 2, 0]}

        knn = KNN(parameters={"n_neighbors": 2})
        knn.fit(EncodedData(sparse.csr_matrix(x), labels=y), "test2")

        test_x = np.array([[0, 1, 0], [1, 0, 0]])
        y = knn.predict(EncodedData(sparse.csr_matrix(test_x)), "test2")

        self.assertTrue(len(y["test2"]) == 2)
        self.assertTrue(y["test2"][1] in [0, 1, 2])

    def test_fit_by_cross_validation(self):
        x = EncodedData(
            np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]]),
            labels={"test1": [1, 0, 2, 0, 1, 0, 2, 0], "test2": [1, 0, 2, 0, 1, 0, 2, 0]})

        knn = KNN(parameters={"n_neighbors": 2})
        knn.fit_by_cross_validation(x, number_of_splits=2, label_name="test1")

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        knn = KNN()
        knn.fit(EncodedData(sparse.csr_matrix(x), y), "default")

        path = EnvironmentSettings.root_path + "test/tmp/knn/"

        knn.store(path, ["f1", "f2", "f3"])
        self.assertTrue(os.path.isfile(path + "knn.pickle"))

        with open(path + "knn.pickle", "rb") as file:
            knn2 = pickle.load(file)

        self.assertTrue(isinstance(knn2["default"], KNeighborsClassifier))

        shutil.rmtree(path)

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        knn = KNN()
        knn.fit(EncodedData(sparse.csr_matrix(x), y), "default")

        path = EnvironmentSettings.root_path + "test/tmp/knn2/"
        PathBuilder.build(path)

        with open(path + "knn.pickle", "wb") as file:
            pickle.dump(knn.get_model(), file)

        knn2 = KNN()
        knn2.load(path)

        self.assertTrue(isinstance(knn2.get_model()["default"], KNeighborsClassifier))

        shutil.rmtree(path)

