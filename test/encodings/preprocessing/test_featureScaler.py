import shutil
from unittest import TestCase

import numpy as np

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestFeatureScaler(TestCase):
    def test_standard_scale(self):
        path = EnvironmentSettings.tmp_test_path / "featurescaler/"
        PathBuilder.build(path)

        feature_matrix = np.array([[0, 2, 3], [0, 0.1, 1], [0, -2, 1]])
        scaled_feature_matrix = FeatureScaler.standard_scale(path / "scaler.pkl", feature_matrix)

        self.assertEqual((3, 3), scaled_feature_matrix.shape)
        self.assertTrue(all(scaled_feature_matrix[i, 0] == 0 for i in range(3)))
        self.assertTrue(all(scaled_feature_matrix[i, 1] != 0 for i in range(3)))

        shutil.rmtree(path)

    def test_normalize(self):
        path = EnvironmentSettings.tmp_test_path / "featurescalernormalize/"
        PathBuilder.build(path)

        feature_matrix = np.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
        norm_feature_matrix = FeatureScaler.normalize(feature_matrix, NormalizationType.L2)
        expected_norm_feature_matrix = np.array([[0.8, 0.2, 0.4, 0.4], [0.1, 0.3, 0.9, 0.3], [0.5, 0.7, 0.5, 0.1]])

        np.testing.assert_array_almost_equal(norm_feature_matrix, expected_norm_feature_matrix)

        shutil.rmtree(path)
