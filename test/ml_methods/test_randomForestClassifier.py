import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier as RFC

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.RandomForestClassifier import RandomForestClassifier
from immuneML.util.PathBuilder import PathBuilder


class TestRandomForestClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        rfc = RandomForestClassifier()
        rfc.fit(EncodedData(sparse.csr_matrix(x), y), Label("default", [1, 0, 2]))

    def test_predict(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        rfc = RandomForestClassifier()
        rfc.fit(EncodedData(x, y), Label("default", [1, 0, 2]))

        test_x = np.array([[0, 1, 0], [1, 0, 0]])
        y = rfc.predict(EncodedData(test_x), Label("default", [1, 0, 2]))["default"]

        self.assertTrue(len(y) == 2)
        self.assertTrue(y[0] in [0, 1, 2])
        self.assertTrue(y[1] in [0, 1, 2])

    def test_fit_by_cross_validation(self):
        x = EncodedData(sparse.csr_matrix(
            np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])),
            labels={"t1": [1, 0, 2, 0, 1, 0, 2, 0], "t2": [1, 0, 2, 0, 1, 0, 2, 0]})

        rfc = RandomForestClassifier()
        rfc.fit_by_cross_validation(x, number_of_splits=2, label=Label("t2", [1, 0, 2]), optimization_metric="balanced_accuracy")

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        rfc = RandomForestClassifier()
        rfc.fit(EncodedData(x, y), Label("default", [1, 0, 2]))

        path = EnvironmentSettings.root_path / "test/tmp/rfc/"

        rfc.store(path)
        self.assertTrue(os.path.isfile(path / "random_forest_classifier.pickle"))

        with open(path / "random_forest_classifier.pickle", "rb") as file:
            rfc2 = pickle.load(file)

        self.assertTrue(isinstance(rfc2, RFC))

        shutil.rmtree(path)

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        rfc = RandomForestClassifier()
        rfc.fit(EncodedData(x, y), Label("default", [1, 0, 2]))

        path = EnvironmentSettings.root_path / "test/tmp/rfc2/"
        PathBuilder.build(path)

        with open(path / "random_forest_classifier.pickle", "wb") as file:
            pickle.dump(rfc.model, file)

        rfc2 = RandomForestClassifier()
        rfc2.load(path)

        self.assertTrue(isinstance(rfc2.model, RFC))

        shutil.rmtree(path)

