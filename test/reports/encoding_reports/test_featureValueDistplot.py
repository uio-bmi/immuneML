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
from source.reports.encoding_reports.FeatureValueDistplot import FeatureValueDistplot


class TestDistributions(TestCase):

    def _create_dummy_encoded_data(self, path):
        n_donors = 8
        n_features = 300
        n_timepoints = 2
        n_examples = n_donors * n_timepoints
        diseased_donors = range(0, 4)

        donors = [donor for donor in range(n_donors) for timepoint in range(n_timepoints)]
        timepoints = [timepoint for donor in range(n_donors) for timepoint in range(n_timepoints)]
        disease_statuses = [donor in diseased_donors for donor in donors]

        kmers = [''.join(random.choices(string.ascii_uppercase, k=3)) for i in range(n_features)]

        encoded_data = {
            'examples': sparse.csr_matrix(
                np.random.normal(50, 10, n_examples * n_features).reshape((n_examples, n_features))),
            'example_ids': [i for i in range(n_examples)],
            'labels': {
                "donor": np.array([f"donor {i}" for i in donors]),
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
                                                      "grouping_label": "donor",
                                                      "color_label": "disease_status",
                                                      "row_grouping_label": "timepoint",
                                                      "distribution_plot_type": "SINA"})

        report.result_name = "sina"
        report.generate()

        # density-like plots (ridge, density) have to work regardless of what 'color_label' is set to
        report = FeatureValueDistplot.build_object(**{"dataset": dataset,
                                                      "result_path": path,
                                                      "grouping_label": "donor",
                                                      "row_grouping_label": "timepoint",
                                                      "distribution_plot_type": "RIDGE"})

        report.result_name = "ridge"
        report.generate()

        report = FeatureValueDistplot.build_object(**{"dataset": dataset,
                                                      "result_path": path,
                                                      "grouping_label": "donor",
                                                      "color_label": "illegal_label",
                                                      "row_grouping_label": "timepoint",
                                                      "distribution_plot_type": "DENSITY"})

        report.result_name = "density"
        report.generate()

        shutil.rmtree(path)
