import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
from sklearn.svm import LinearSVC

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.SVM import SVM
from immuneML.util.PathBuilder import PathBuilder


class TestSVM(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        svm = SVM()
        svm.fit(EncodedData(x, y), 'default')

    def test_predict(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"test": np.array([1, 0, 2, 0])}

        svm = SVM()
        svm.fit(EncodedData(x, y), "test")

        test_x = np.array([[0, 1, 0], [1, 0, 0]])
        y = svm.predict(EncodedData(test_x), 'test')["test"]

        self.assertTrue(len(y) == 2)
        self.assertTrue(y[0] in [0, 1, 2])
        self.assertTrue(y[1] in [0, 1, 2])

    def test_fit_by_cross_validation(self):
        x = EncodedData(np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]]),
                        {"t1": [1, 0, 2, 0, 1, 0, 2, 0], "t2": [1, 0, 2, 0, 1, 0, 2, 0]})

        svm = SVM()
        svm.fit_by_cross_validation(x, number_of_splits=2, label_name="t1")

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array(['a', "b", "c", "a"])}

        svm = SVM()
        svm.fit(EncodedData(x, y), 'default')

        path = EnvironmentSettings.root_path / "test/tmp/svm/"

        svm.store(path)
        self.assertTrue(os.path.isfile(path / "svm.pickle"))

        with open(path / "svm.pickle", "rb") as file:
            svm2 = pickle.load(file)

        self.assertTrue(isinstance(svm2["default"], LinearSVC))

        shutil.rmtree(path)

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        svm = SVM()
        svm.fit(EncodedData(x, y), 'default')

        path = EnvironmentSettings.root_path / "test/tmp/svm2/"
        PathBuilder.build(path)

        with open(path / "svm.pickle", "wb") as file:
            pickle.dump(svm.get_model(), file)

        svm2 = SVM()
        svm2.load(path)

        self.assertTrue(isinstance(svm2.get_model()["default"], LinearSVC))

        shutil.rmtree(path)
