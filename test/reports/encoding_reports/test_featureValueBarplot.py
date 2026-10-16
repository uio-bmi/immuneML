import os
import random
import shutil
import string
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy import sparse

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.FeatureValueBarplot import FeatureValueBarplot
from immuneML.util.PathBuilder import PathBuilder


class TestFeatureValueBarplot(TestCase):

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

        metadata = pd.DataFrame({"patient": np.array([i for i in range(n_subjects)]),
                                 "disease": np.array(["disease 1"] * int(n_subjects / 2) + ["disease 2"] * int(n_subjects / 2)),
                                 "timepoint": np.array(["timepoint 1", "timepoint 2"] * int(n_subjects / 2))
                                 })

        metadata.to_csv(metadata_filepath, index=False)

        dataset = RepertoireDataset(encoded_data=EncodedData(**encoded_data), metadata_file=metadata_filepath)

        return dataset

    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / "featurevaluebarplot/"
        PathBuilder.remove_old_and_build(path)

        dataset = self._create_dummy_encoded_data(path)

        report = FeatureValueBarplot.build_object(**{"dataset": dataset, 'plot_all_features': True,
                                                     "result_path": path, 'plot_top_n': 10, 'plot_bottom_n': 5,
                                                     "column_grouping_label": None, 'error_function': 'sem',
                                                     "row_grouping_label": "timepoint",
                                                     "color_grouping_label": "disease"})

        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertEqual(result.output_figures[0].path, path / "feature_value_barplot_all.html")
        self.assertEqual(result.output_tables[0].path, path / "feature_value_barplot_all.csv")

        # feature values are aggregated per feature and per group, not listed per example
        content = pd.read_csv(path / "feature_values.csv")
        self.assertSetEqual(set(content.columns), {"feature", "valuesem", "valuemean", "disease", "timepoint", "sequence"})
        self.assertEqual(len(content), 30 * 4)

        examples = dataset.encoded_data.examples.toarray()
        metadata = pd.read_csv(path / "metadata.csv")
        group_rows = (metadata["disease"] == "disease 1") & (metadata["timepoint"] == "timepoint 1")
        expected_mean = examples[group_rows.values].mean(axis=0)
        reported = content[(content["disease"] == "disease 1") & (content["timepoint"] == "timepoint 1")]
        self.assertListEqual(list(reported["feature"]), list(dataset.encoded_data.feature_names))
        self.assertTrue(np.allclose(reported["valuemean"].values, expected_mean))

        # report should succeed to build_from_objects but check_prerequisites should be false when data is not encoded
        report = FeatureValueBarplot.build_object(**{"dataset": RepertoireDataset(),
                                                     "result_path": path,
                                                     "column_grouping_label": None,
                                                     "row_grouping_label": None,
                                                     "color_grouping_label": None})

        self.assertFalse(report.check_prerequisites())

        shutil.rmtree(path)
