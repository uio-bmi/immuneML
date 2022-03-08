import os
import pickle
import shutil
from unittest import TestCase

import dill
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.TCRdistClassifier import TCRdistClassifier
from immuneML.util.PathBuilder import PathBuilder


class TestTCRdistClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _prepare_data(self):
        x = np.array([[0., 1., 2., 3.],
                      [1., 0., 1., 2.],
                      [2., 1., 0., 1.],
                      [3., 2., 1., 0.]])
        y = {"test": np.array([0, 0, 1, 1])}

        return x, y, EncodedData(examples=x, labels=y)

    def test_fit(self):
        x, y, encoded_data = self._prepare_data()
        knn = TCRdistClassifier(percentage=0.75)
        knn.fit(encoded_data, Label("test"), cores_for_training=4)
        predictions = knn.predict(encoded_data, Label("test"))
        self.assertTrue(np.array_equal(y["test"], predictions["test"]))

        encoded_data.examples = np.array([[1.1, 0.1, 0.9, 1.9]])
        predictions = knn.predict(encoded_data, Label("test"))
        self.assertTrue(np.array_equal([0], predictions["test"]))

    def test_store(self):
        x, y, encoded_data = self._prepare_data()

        cls = TCRdistClassifier(0.75)
        cls.fit(encoded_data, label=Label("test"), cores_for_training=4)

        path = EnvironmentSettings.root_path / "test/tmp/tcrdist_classifier/"

        cls.store(path)
        self.assertTrue(os.path.isfile(path / "tcrdist_classifier.pickle"))

        with open(path / "tcrdist_classifier.pickle", "rb") as file:
            cls2 = pickle.load(file)

        self.assertTrue(isinstance(cls2, KNeighborsClassifier))

        shutil.rmtree(path)

    def test_load(self):
        x, y, encoded_data = self._prepare_data()

        cls = TCRdistClassifier(0.75)
        cls.fit(encoded_data, label=Label("test"), cores_for_training=4)

        path = PathBuilder.build(EnvironmentSettings.root_path / "test/tmp/tcrdist_classifier_load/")

        with open(path / "tcrdist_classifier.pickle", "wb") as file:
            dill.dump(cls.model, file)

        cls2 = TCRdistClassifier(percentage=1.)
        cls2.load(path)

        self.assertTrue(isinstance(cls2.model, KNeighborsClassifier))
        self.assertTrue(isinstance(cls2, TCRdistClassifier))
        self.assertEqual(3, cls2.model.n_neighbors)

        shutil.rmtree(path)
