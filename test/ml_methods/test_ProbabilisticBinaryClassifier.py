import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.ProbabilisticBinaryClassifier import ProbabilisticBinaryClassifier
from immuneML.util.PathBuilder import PathBuilder


class TestProbabilisticBinaryClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def train_classifier(self):
        classifier = ProbabilisticBinaryClassifier(100, 0.1)

        X = np.array([[3, 4], [1, 7], [5, 7], [3, 8]])
        y = {"cmv": [True, False, True, False]}

        classifier.fit(EncodedData(X, y), Label("cmv", [True, False]))

        return classifier

    def test_fit(self):

        classifier = self.train_classifier()

        predictions = classifier.predict(EncodedData(np.array([[6, 7], [1, 6]])), Label("cmv", [True, False]))
        proba_predictions = classifier.predict_proba(EncodedData(np.array([[6, 7], [1, 6]])), Label("cmv", [True, False]))

        self.assertEqual([True, False], predictions["cmv"])

        self.assertTrue((proba_predictions["cmv"][True] <= 1.0).all() and (proba_predictions["cmv"][True] >= 0.0).all())
        self.assertTrue((proba_predictions["cmv"][False] <= 1.0).all() and (proba_predictions["cmv"][False] >= 0.0).all())

        self.assertListEqual(list(proba_predictions["cmv"][True] > 0.5), [pred == True for pred in list(predictions["cmv"])])

    def test_store(self):

        classifier = self.train_classifier()

        path = EnvironmentSettings.tmp_test_path / "probabilistic_binary_classifier/"
        PathBuilder.build(path)

        classifier.store(path=path)

        self.assertTrue(os.path.isfile(path / "probabilistic_binary_classifier.pickle"))
        self.assertTrue(os.path.isfile(path / "probabilistic_binary_classifier.yaml"))

        shutil.rmtree(path)

