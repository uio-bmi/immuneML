import os
from unittest import TestCase

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.encodings.abundance_encoding.AbundanceEncoderHelper import AbundanceEncoderHelper
from immuneML.environment.Constants import Constants


class TestAbundanceEncoderHelper(TestCase):
    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_find_sequence_p_values_with_fisher(self):
        sequence_presence_matrix = np.array([[1, 0, 0, 0],
                                             [1, 0, 1, 1],
                                             [1, 1, 1, 1],
                                             [0, 0, 0, 1],
                                             [0, 0, 1, 0]])

        is_positive_class = np.array([True, False, False, True])

        import fisher
        contingency_table = AbundanceEncoderHelper._get_contingency_table(sequence_presence_matrix, is_positive_class)
        p_values = AbundanceEncoderHelper._find_sequence_p_values_with_fisher(contingency_table, fisher)

        expected = [2, 0.5, 1.0, 2, 2]

        for real, expected in zip(p_values, expected):
            self.assertAlmostEqual(real, expected, places=7)

    def test_build_abundance_matrix(self):
        sequence_presence_matrix = np.array([[0, 1], [1, 1], [1, 0]])
        matrix_repertoire_ids = np.array(["r1", "r2"])
        dataset_repertoire_ids = ["r2"]
        sequence_p_values_indices = [True, False, True]

        abundance_matrix = AbundanceEncoderHelper.build_abundance_matrix(sequence_presence_matrix,
                                                                         matrix_repertoire_ids,
                                                                         dataset_repertoire_ids,
                                                                         sequence_p_values_indices)

        expected = np.array([[1., 2.]])

        self.assertTrue((abundance_matrix == expected).all())
