import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans as SklearnKMeans

from immuneML.IO.ml_method.MLMethodConfiguration import MLMethodConfiguration
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.Clustering.KMeans import KMeans
from immuneML.util.PathBuilder import PathBuilder


class TestUnsupervisedSklearnMethod(TestCase):
    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
        y = {"default": np.array([1, 0, 2, 0])}

        km = KMeans()
        km.fit(EncodedData(examples=sparse.csr_matrix(x), labels=y))

        path = EnvironmentSettings.root_path / "test/tmp/loadtestunsupervisedsklearn/"
        PathBuilder.build(path)

        with open(path / "k_means.pickle", "wb") as file:
            pickle.dump(km.model, file)

        config = MLMethodConfiguration()
        config.labels_with_values = {"default": [0, 1, 2]}
        config.store(path / "config.json")

        km2 = KMeans()
        km2.load(path)

        self.assertTrue(isinstance(km2.model, SklearnKMeans))

        shutil.rmtree(path)

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])

        km = KMeans()
        km._fit(sparse.csr_matrix(x))

        path = EnvironmentSettings.root_path / "test/tmp/storeunsupervisedsklearn/"

        km.store(path)
        self.assertTrue(os.path.isfile(path / "k_means.pickle"))

        with open(path / "k_means.pickle", "rb") as file:
            km2 = pickle.load(file)

        self.assertTrue(isinstance(km2, SklearnKMeans))

        shutil.rmtree(path)

    def test_store_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])

        km = KMeans()
        km._fit(sparse.csr_matrix(x))

        path = EnvironmentSettings.root_path / "test/tmp/store_load_unsupervisedsklearn/"
        details_path = EnvironmentSettings.root_path / "test/tmp/store_load_unsupervisedsklearn/details.yaml"

        km.store(path=path, details_path=details_path)

        km2 = KMeans()
        km2.load(path=path, details_path=details_path)

        shutil.rmtree(path)

