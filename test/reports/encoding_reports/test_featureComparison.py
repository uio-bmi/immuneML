import os
import random
import shutil
import string
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy import sparse

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.FeatureComparison import FeatureComparison
from immuneML.reports.encoding_reports.FeatureDistribution import FeatureDistribution
from immuneML.util.PathBuilder import PathBuilder


class TestFeatureComparison(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_encoded_data(self, path):
        n_subjects = 50
        n_features = 30

        kmers = [''.join(random.choices(string.ascii_uppercase, k=3)) for i in range(n_features)]

        encoded_data = {
            'examples': sparse.csr_matrix(
                np.random.normal(50, 10, n_subjects * n_features).reshape((n_subjects, n_features))),
            'example_ids': [''.join(random.choices(string.ascii_uppercase, k=4)) for i in range(n_subjects)],
            'labels': {
            },
            'feature_names': kmers,
            'feature_annotations': pd.DataFrame({
                "sequence": kmers
            }),
            'encoding': "random"
        }

        metadata_filepath = path / "metadata.csv"

        metadata = pd.DataFrame({"patient": np.array([i % 2 == 0 for i in range(n_subjects)])})
        metadata.to_csv(metadata_filepath, index=False)

        dataset = RepertoireDataset(encoded_data=EncodedData(**encoded_data), metadata_file=metadata_filepath)

        return dataset

    def test_generate(self):
        path = EnvironmentSettings.root_path / "test/tmp/featurecomparison/"
        PathBuilder.build(path)

        dataset = self._create_dummy_encoded_data(path)

        report = FeatureComparison.build_object(**{"dataset": dataset,
                                                     "result_path": path,
                                                     "comparison_label": "patient",
                                                   "keep_fraction": 0.2,
                                                   "log_scale": True})

        self.assertTrue(report.check_prerequisites())

        result = report.generate_report()

        self.assertIsInstance(result, ReportResult)

        self.assertEqual(result.output_figures[0].path, path / "feature_comparison.html")
        self.assertEqual(result.output_tables[0].path, path / "feature_values.csv")

        content = pd.read_csv(path / "feature_values.csv")
        self.assertListEqual(list(content.columns),
                             ["patient", "example_id", "sequence", "feature", "value"])

        # report should succeed to build_from_objects but check_prerequisites should be false when data is not encoded
        report = FeatureDistribution.build_object(**{"dataset": RepertoireDataset(),
                                                     "result_path": path})

        self.assertFalse(report.check_prerequisites())

        shutil.rmtree(path)
