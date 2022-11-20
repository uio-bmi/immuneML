import os
from unittest import TestCase

import numpy as np

from immuneML.analysis.entropy_calculations.EntropyCalculator import EntropyCalculator
from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants


class TestEvennessCalculator(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_renyi_entropy(self):
        x = np.array([100, 50, 1])
        alpha = 1
        result = EntropyCalculator.renyi_entropy(x, alpha)
        self.assertAlmostEqual(result, 0.6721264005883254)

        x = np.array([100, 50, 1])
        alpha = 0
        result = EntropyCalculator.renyi_entropy(x, alpha)
        self.assertAlmostEqual(result, 1.0986122886681098)

        x = np.array([100, 50, 1])
        alpha = 10
        result = EntropyCalculator.renyi_entropy(x, alpha)
        self.assertAlmostEqual(result, 0.4577911580328081)
