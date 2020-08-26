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
from source.reports.encoding_reports.FeatureValueDistplot import FeatureValueDistplot


class TestDistributions(TestCase):

    def _create_dummy_encoded_data(self, path):
        n_subjects = 8
        n_features = 300
        n_timepoints = 2
        n_examples = n_subjects * n_timepoints
        diseased_subjects = range(0, 4)

        subjects = [subject for subject in range(n_subjects) for timepoint in range(n_timepoints)]
        timepoints = [timepoint for subject in range(n_subjects) for timepoint in range(n_timepoints)]
        disease_statuses = [subject in diseased_subjects for subject in subjects]

        kmers = [''.join(random.choices(string.ascii_uppercase, k=3)) for i in range(n_features)]

        encoded_data = {
            'examples': sparse.csr_matrix(
                np.random.normal(50, 10, n_examples * n_features).reshape((n_examples, n_features))),
            'example_ids': [i for i in range(n_examples)],
            'labels': {
                "subject_id": np.array([f"subject {i}" for i in subjects]),
                "disease_status": np.array([f"disease: {i}" for i in disease_statuses]),
                "timepoint": np.array([f"timepoint {i}" for i in timepoints])
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
        path = EnvironmentSettings.root_path + "test/tmp/featurevaluedistplot/"

        dataset = self._create_dummy_encoded_data(path)

        report = FeatureValueDistplot.build_object(**{"dataset": dataset,
                                                      "result_path": path,
                                                      "grouping_label": "subject_id",
                                                      "color_label": "disease_status",
                                                      "row_grouping_labels": "timepoint",
                                                      "distribution_plot_type": "SINA"})

        report.result_name = "sina"
        result = report.generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(result.output_figures[0].path, path+"sina.pdf")
        self.assertEqual(result.output_tables[0].path, path+"sina.csv")

        # density-like plots (ridge, density) have to work regardless of what 'color_label' is set to
        report = FeatureValueDistplot.build_object(**{"dataset": dataset,
                                                      "result_path": path,
                                                      "grouping_label": "subject_id",
                                                      "row_grouping_labels": "timepoint",
                                                      "distribution_plot_type": "RIDGE"})

        report.result_name = "ridge"
        result = report.generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(result.output_figures[0].path, path + "ridge.pdf")
        self.assertEqual(result.output_tables[0].path, path + "ridge.csv")

        report = FeatureValueDistplot.build_object(**{"dataset": dataset,
                                                      "result_path": path,
                                                      "grouping_label": "subject_id",
                                                      "color_label": "illegal_label",
                                                      "row_grouping_labels": "timepoint",
                                                      "distribution_plot_type": "DENSITY"})

        report.result_name = "density"
        result = report.generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(result.output_figures[0].path, path + "density.pdf")
        self.assertEqual(result.output_tables[0].path, path + "density.csv")

        shutil.rmtree(path)
