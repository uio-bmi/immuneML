from unittest import TestCase

import numpy as np

from source.ml_methods.RankClassifier import RankClassifier


class TestRankClassifier(TestCase):
    def test__fit_for_label(self):
        X = np.array([[5], [4], [3], [2], [1]])
        y = {"l1": np.array([1, 1, 0, 1, 0])}

        rank_classifier = RankClassifier()
        rank_classifier._fit_for_label("l1", X, y)

        self.assertEqual(1.5, rank_classifier._models["l1"]["threshold"])

    def test_fit(self):
        X = np.array([[4], [5], [1], [2], [3]])
        y = {"l1": np.array([True, True, False, True, False]), "l2": np.array([0, 0, 0, 1, 0, 1])}

        rank_cls = RankClassifier()
        rank_cls.fit(X, y, ["l1", "l2"])
        self.assertEqual(1.5, rank_cls._models["l1"]["threshold"])

    def test_predict(self):
        X = np.array([[5], [4], [3], [2], [1]])
        y = {"l1": np.array([True, True, False, True, False]), "l2": np.array([0, 0, 0, 1, 0, 1])}

        rank_cls = RankClassifier()
        rank_cls.fit(X, y, ["l1", "l2"])
        predictions = rank_cls.predict([[6], [3.4], [0.001]], ["l1", "l2"])
        self.assertTrue("l1" in predictions and "l2" in predictions)
        self.assertEqual(np.dtype('bool'), predictions["l1"].dtype)