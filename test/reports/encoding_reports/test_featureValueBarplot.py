import random
import shutil
import string
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy import sparse

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.ReportResult import ReportResult
from source.reports.encoding_reports.FeatureValueBarplot import FeatureValueBarplot


class TestFeatureValueBarplot(TestCase):

    def _create_dummy_encoded_data(self, path):
        n_donors = 50
        n_features = 30

        kmers = [''.join(random.choices(string.ascii_uppercase, k=3)) for i in range(n_features)]

        encoded_data = {
            'examples': sparse.csr_matrix(
                np.random.normal(50, 10, n_donors * n_features).reshape((n_donors, n_features))),
            'example_ids': [''.join(random.choices(string.ascii_uppercase, k=4)) for i in range(n_donors)],
            'labels': {
                "patient": np.array([i for i in range(n_donors)]),
                "disease": np.array(["disease 1"] * int(n_donors / 2) + ["disease 2"] * int(n_donors / 2)),
                "timepoint": np.array(["timepoint 1", "timepoint 2"] * int(n_donors / 2))
            },
            'feature_names': kmers,
            'feature_annotations': pd.DataFrame({
                "sequence": kmers
            }),
            'encoding': "random"
        }

        dataset = RepertoireDataset(encoded_data=EncodedData(**encoded_data))

        return dataset

    def test_generate(self):
        path = EnvironmentSettings.root_path + "test/tmp/featurevaluebarplot/"

        dataset = self._create_dummy_encoded_data(path)

        report = FeatureValueBarplot.build_object(**{"dataset": dataset,
                                          "result_path": path,
                                          "errorbar_meaning": "STANDARD_ERROR",
                                          "column_grouping_labels": "disease",
                                          "row_grouping_labels": "timepoint",
                                          "color_grouping_label": "disease"})

        self.assertTrue(report.check_prerequisites())

        result = report.generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(result.output_figures[0].path, path+"feature_values.pdf")
        self.assertEqual(result.output_tables[0].path, path+"feature_values.csv")

        content = pd.read_csv(f"{path}/feature_values.csv")
        self.assertListEqual(list(content.columns), ["patient", "disease", "timepoint", "example_id", "sequence", "feature", "value"])

        # report should succeed to build but check_prerequisites should be false when data is not encoded
        report = FeatureValueBarplot.build_object(**{"dataset": RepertoireDataset(),
                                            "result_path": path,
                                            "errorbar_meaning": "STANDARD_ERROR",
                                            "column_grouping_labels": None,
                                            "row_grouping_labels": None,
                                            "color_grouping_label": None})

        self.assertFalse(report.check_prerequisites())

        shutil.rmtree(path)
