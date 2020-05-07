import os
from unittest import TestCase

import numpy as np

from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.ml_methods.ProbabilisticBinaryClassifier import ProbabilisticBinaryClassifier


class TestProbabilisticBinaryClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):

        classifier = ProbabilisticBinaryClassifier(100, 0.1)

        X = np.array([[3, 4], [1, 7], [5, 7], [3, 8]])
        y = {"cmv": [1, 0, 1, 0]}

        classifier.fit(X, y, ["cmv"])

        predictions = classifier.predict([[6, 7], [1, 6]], ["cmv"])

        self.assertEqual([1, 0], predictions)
