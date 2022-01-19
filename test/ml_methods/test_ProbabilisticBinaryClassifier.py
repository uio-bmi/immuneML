import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.ProbabilisticBinaryClassifier import ProbabilisticBinaryClassifier
from immuneML.util.PathBuilder import PathBuilder


class TestProbabilisticBinaryClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def train_classifier(self):
        classifier = ProbabilisticBinaryClassifier(100, 0.1)

        X = np.array([[3, 4], [1, 7], [5, 7], [3, 8]])
        y = {"cmv": [True, False, True, False]}

        classifier.fit(EncodedData(X, y), Label("cmv"))

        return classifier

    def test_fit(self):

        classifier = self.train_classifier()

        predictions = classifier.predict(EncodedData(np.array([[6, 7], [1, 6]])), Label("cmv"))
        proba_predictions = classifier.predict_proba(EncodedData(np.array([[6, 7], [1, 6]])), Label("cmv"))

        self.assertEqual([True, False], predictions["cmv"])
        self.assertTrue(proba_predictions["cmv"][0, 1] > proba_predictions["cmv"][0, 0])
        self.assertTrue(proba_predictions["cmv"][1, 0] > proba_predictions["cmv"][1, 1])
        self.assertTrue((proba_predictions["cmv"] <= 1.0).all() and (proba_predictions["cmv"] >= 0.0).all())

    def test_store(self):

        classifier = self.train_classifier()

        path = EnvironmentSettings.tmp_test_path / "probabilistic_binary_classifier/"
        PathBuilder.build(path)

        classifier.store(path=path, feature_names=["k_i", "n_i"])

        self.assertTrue(os.path.isfile(path / "probabilistic_binary_classifier.pickle"))
        self.assertTrue(os.path.isfile(path / "probabilistic_binary_classifier.yaml"))

        shutil.rmtree(path)

