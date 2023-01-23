import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering as SklearnSpectral

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.Clustering.Spectral import Spectral
from immuneML.util.PathBuilder import PathBuilder


class TestSpectral(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])

        method = Spectral()
        method.fit(EncodedData(examples=sparse.csr_matrix(x)))

        self.assertTrue(method.model.labels_ is not None)

    def test_store(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])

        method = Spectral()
        method.fit(EncodedData(examples=sparse.csr_matrix(x)))

        path = EnvironmentSettings.root_path / "test/tmp/method/"

        method.store(path, ["f1", "f2", "f3"])
        pickle_file_path = path / "spectral.pickle"
        self.assertTrue(os.path.isfile(str(pickle_file_path)))

        with pickle_file_path.open("rb") as file:
            method2 = pickle.load(file)

        self.assertTrue(isinstance(method2, SklearnSpectral))

        shutil.rmtree(path)

    def test_load(self):
        x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])

        method = Spectral()
        method.fit(EncodedData(sparse.csr_matrix(x)))

        path = EnvironmentSettings.root_path / "test/tmp/method2/"
        PathBuilder.build(path)

        pickle_file_path = path / "spectral.pickle"
        with pickle_file_path.open("wb") as file:
            pickle.dump(method.model, file)

        method2 = Spectral()
        method2.load(path)

        self.assertTrue(isinstance(method2.model, SklearnSpectral))

        shutil.rmtree(path)

