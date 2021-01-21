import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from source.caching.CacheType import CacheType
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.LogisticRegression import LogisticRegression
from source.util.PathBuilder import PathBuilder


class TestLogisticRegression(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"test": np.array([1, 0, 2, 0])}

        lr = LogisticRegression()
        lr.fit(EncodedData(x, y), "test")

    def test_predict(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"test1": [1, 0, 2, 0], "test2": [1, 0, 2, 0]}

        lr = LogisticRegression()
        lr.fit(EncodedData(x, y), "test2")

        test_x = np.array([[0, 1, 0], [1, 0, 0]])
        y = lr.predict(EncodedData(test_x), "test2")

        self.assertTrue(len(y["test2"]) == 2)
        self.assertTrue(y["test2"][1] in [0, 1, 2])

    def test_fit_by_cross_validation(self):
        x = EncodedData(
            np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]]),
            {"test1": [1, 0, 2, 0, 1, 0, 2, 0], "test2": [1, 0, 2, 0, 1, 0, 2, 0]})

        lr = LogisticRegression()
        lr.fit_by_cross_validation(x, number_of_splits=2, label_name="test2")

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        lr = LogisticRegression()
        lr.fit(EncodedData(x, y), 'default')

        path = EnvironmentSettings.root_path / "test/tmp/lr/"

        lr.store(path, ["f1", "f2", "f3"])
        self.assertTrue(os.path.isfile(path / "logistic_regression.pickle"))

        with open(path / "logistic_regression.pickle", "rb") as file:
            lr2 = pickle.load(file)

        self.assertTrue(isinstance(lr2["default"], SklearnLogisticRegression))

        shutil.rmtree(path)

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        lr = LogisticRegression()
        lr.fit(EncodedData(x, y), 'default')

        path = EnvironmentSettings.root_path / "test/tmp/lr2/"
        PathBuilder.build(path)

        with open(path / "logistic_regression.pickle", "wb") as file:
            pickle.dump(lr.get_model(), file)

        lr2 = LogisticRegression()
        lr2.load(path)

        self.assertTrue(isinstance(lr2.get_model()["default"], SklearnLogisticRegression))

        shutil.rmtree(path)

