import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier as RFC

from source.caching.CacheType import CacheType
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.RandomForestClassifier import RandomForestClassifier
from source.util.PathBuilder import PathBuilder


class TestRandomForestClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        rfc = RandomForestClassifier()
        rfc.fit(EncodedData(sparse.csr_matrix(x)), y)

    def test_predict(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        rfc = RandomForestClassifier()
        rfc.fit(EncodedData(x), y)

        test_x = np.array([[0, 1, 0], [1, 0, 0]])
        y = rfc.predict(EncodedData(test_x))["default"]

        self.assertTrue(len(y) == 2)
        self.assertTrue(y[0] in [0, 1, 2])
        self.assertTrue(y[1] in [0, 1, 2])

    def test_fit_by_cross_validation(self):
        x = EncodedData(sparse.csr_matrix(
            np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])))
        y = {"t1": [1, 0, 2, 0, 1, 0, 2, 0], "t2": [1, 0, 2, 0, 1, 0, 2, 0]}

        rfc = RandomForestClassifier()
        rfc.fit_by_cross_validation(x, y, number_of_splits=2, label_names=["t1", "t2"])

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        rfc = RandomForestClassifier()
        rfc.fit(EncodedData(x), y)

        path = EnvironmentSettings.root_path + "test/tmp/rfc/"

        rfc.store(path)
        self.assertTrue(os.path.isfile(path + "random_forest_classifier.pickle"))

        with open(path + "random_forest_classifier.pickle", "rb") as file:
            rfc2 = pickle.load(file)

        self.assertTrue(isinstance(rfc2["default"], RFC))

        shutil.rmtree(path)

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        rfc = RandomForestClassifier()
        rfc.fit(EncodedData(x), y)

        path = EnvironmentSettings.root_path + "test/tmp/rfc2/"
        PathBuilder.build(path)

        with open(path + "random_forest_classifier.pickle", "wb") as file:
            pickle.dump(rfc.get_model(), file)

        rfc2 = RandomForestClassifier()
        rfc2.load(path)

        self.assertTrue(isinstance(rfc2.get_model()["default"], RFC))

        shutil.rmtree(path)

