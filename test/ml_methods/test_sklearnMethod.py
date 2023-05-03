import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from immuneML.IO.ml_method.MLMethodConfiguration import MLMethodConfiguration
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.KNN import KNN
from immuneML.ml_methods.SVM import SVM
from immuneML.util.PathBuilder import PathBuilder


class TestSklearnMethod(TestCase):
    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        knn = KNN()
        knn.fit(EncodedData(examples=sparse.csr_matrix(x), labels=y), Label("default", [1, 0, 2]))

        path = EnvironmentSettings.root_path / "test/tmp/loadtestsklearn/"
        PathBuilder.build(path)

        with open(path / "knn.pickle", "wb") as file:
            pickle.dump(knn.model, file)

        config = MLMethodConfiguration()
        config.labels_with_values = {"default": [0, 1, 2]}
        config.store(path / "config.json")

        knn2 = KNN()
        knn2.load(path)

        self.assertTrue(isinstance(knn2.model, KNeighborsClassifier))

        shutil.rmtree(path)

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array(['a', "b", "c", "a"])}

        svm = SVM()
        svm._fit(sparse.csr_matrix(x), y["default"])

        path = EnvironmentSettings.root_path / "test/tmp/storesklearn/"

        svm.store(path)
        self.assertTrue(os.path.isfile(path / "svm.pickle"))

        with open(path / "svm.pickle", "rb") as file:
            svm2 = pickle.load(file)

        self.assertTrue(isinstance(svm2, SVC))

        shutil.rmtree(path)

    def test_store_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array(['a', "b", "c", "a"])}

        svm = SVM()
        svm._fit(sparse.csr_matrix(x), y["default"])

        path = EnvironmentSettings.root_path / "test/tmp/store_load_sklearn/"
        details_path = EnvironmentSettings.root_path / "test/tmp/store_load_sklearn/details.yaml"

        svm.store(path=path, details_path=details_path)

        svm2 = SVM()
        svm2.load(path=path, details_path=details_path)

        shutil.rmtree(path)

