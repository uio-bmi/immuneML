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
from immuneML.environment.Label import Label
from immuneML.ml_methods.SVC import SVC
from immuneML.util.PathBuilder import PathBuilder


class TestSVC(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        svm = SVC()
        svm.fit(EncodedData(x, y), Label("default", [1, 0, 2]))

    def test_predict(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"test": np.array([1, 0, 2, 0])}

        svm = SVC()
        svm.fit(EncodedData(x, y), Label("test", [1, 0, 2]))

        test_x = np.array([[0, 1, 0], [1, 0, 0]])
        y = svm.predict(EncodedData(test_x), Label("test", [1, 0, 2]))["test"]

        self.assertTrue(len(y) == 2)
        self.assertTrue(y[0] in [0, 1, 2])
        self.assertTrue(y[1] in [0, 1, 2])

    def test_fit_by_cross_validation(self):
        x = EncodedData(np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]]),
                        {"t1": [1, 0, 2, 0, 1, 0, 2, 0], "t2": [1, 0, 2, 0, 1, 0, 2, 0]})

        svm = SVC(parameter_grid={"penalty": ["l1"], "dual": [False]})
        svm.fit_by_cross_validation(x, number_of_splits=2, label=Label("t1", [1, 0, 2]))

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array(['a', "b", "c", "a"])}

        svm = SVC()
        svm.fit(EncodedData(x, y), Label("default", ["a", "b", "c"]))

        path = EnvironmentSettings.root_path / "my_svc/"

        svm.store(path)
        self.assertTrue(os.path.isfile(path / "svc.pickle"))

        with open(path / "svc.pickle", "rb") as file:
            svm2 = pickle.load(file)

        self.assertTrue(isinstance(svm2, LinearSVC))

        shutil.rmtree(path)

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        svm = SVC()
        svm.fit(EncodedData(x, y), Label("default", [0, 1, 2]))

        path = EnvironmentSettings.tmp_test_path / "my_svc2/"
        PathBuilder.build(path)

        with open(path / "svc.pickle", "wb") as file:
            pickle.dump(svm.model, file)

        svm2 = SVC()
        svm2.load(path)

        self.assertTrue(isinstance(svm2.model, LinearSVC))

        shutil.rmtree(path)
