import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.LSTM import LSTM
from immuneML.util.PathBuilder import PathBuilder


class TestLogisticRegression(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"test": np.array([1, 0, 2, 0])}

        path = EnvironmentSettings.root_path / "test/tmp/lr/"

        lr = LSTM()
        lr.fit(EncodedData(x, y, info={"length_of_sequence": 21}), result_path=path)

    def test_generate(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"test1": [1, 0, 2, 0], "test2": [1, 0, 2, 0]}

        lstm = LSTM()
        lstm.fit(EncodedData(x, y, info={"length_of_sequence": 21}))

        test_x = np.array([[0, 1, 0], [1, 0, 0]])
        y = lstm.generate(EncodedData(test_x))

        self.assertTrue(len(y["test2"]) == 2)
        self.assertTrue(y["test2"][1] in [0, 1, 2])

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        lstm = LSTM()
        lstm.fit(EncodedData(x, y))

        path = EnvironmentSettings.root_path / "test/tmp/lr/"

        lstm.store(path)
        self.assertTrue(os.path.isfile(path / "logistic_regression.csv"))

        with open(path / "logistic_regression.pickle", "rb") as file:
            lr2 = pickle.load(file)

        self.assertTrue(isinstance(lr2, SklearnLogisticRegression))

        shutil.rmtree(path)

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        lstm = LSTM()
        lstm.fit(EncodedData(x, y))

        path = EnvironmentSettings.root_path / "test/tmp/lr2/"
        PathBuilder.build(path)

        with open(path / "logistic_regression.pickle", "wb") as file:
            pickle.dump(lstm.model, file)

        lstm2 = LSTM()
        lstm2.load(path)

        self.assertTrue(isinstance(lstm2.model, SklearnLogisticRegression))

        shutil.rmtree(path)

