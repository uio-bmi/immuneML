import os
import shutil
from unittest import TestCase

import numpy as np

from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.ProbabilisticBinaryClassifier import ProbabilisticBinaryClassifier
from source.util.PathBuilder import PathBuilder


class TestProbabilisticBinaryClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def train_classifier(self):
        classifier = ProbabilisticBinaryClassifier(100, 0.1)

        X = np.array([[3, 4], [1, 7], [5, 7], [3, 8]])
        y = {"cmv": [1, 0, 1, 0]}

        classifier.fit(X, y, ["cmv"])

        return classifier

    def test_fit(self):

        classifier = self.train_classifier()

        predictions = classifier.predict([[6, 7], [1, 6]], ["cmv"])

        self.assertEqual([1, 0], predictions["cmv"])

    def test_store(self):

        classifier = self.train_classifier()

        path = EnvironmentSettings.tmp_test_path + "probabilistic_binary_classifier/"
        PathBuilder.build(path)

        classifier.store(path=path, feature_names=["k_i", "n_i"])

        self.assertTrue(os.path.isfile(f"{path}probabilistic_binary_classifier.pickle"))
        self.assertTrue(os.path.isfile(f"{path}probabilistic_binary_classifier.yaml"))

        shutil.rmtree(path)

