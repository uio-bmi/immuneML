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

    def _create_two_group_dataset(self, path, examples):
        """Two groups of two examples each over three features, so the divergence is known exactly."""
        encoded_data = {'examples': sparse.csr_matrix(np.array(examples, dtype=float)),
                        'example_ids': ["e1", "e2", "e3", "e4"],
                        'labels': {},
                        'feature_names': ["AAA", "CCC", "DDD"],
                        'encoding': "random"}

        metadata_filepath = path / "metadata.csv"
        pd.DataFrame({"subject": [1, 2, 3, 4],
                      "origin": ["original", "original", "generated", "generated"]}).to_csv(metadata_filepath,
                                                                                            index=False)

        return RepertoireDataset(encoded_data=EncodedData(**encoded_data), metadata_file=metadata_filepath)

    def _build_report(self, dataset, path, **kwargs):
        params = {"dataset": dataset, "result_path": path, "plot_all_features": True,
                  "color_grouping_label": "origin", "row_grouping_label": None, "column_grouping_label": None,
                  "error_function": "std", "compute_jensen_shannon": True}
        params.update(kwargs)
        return FeatureValueBarplot.build_object(**params)

    def test_jensen_shannon_divergence(self):
        path = EnvironmentSettings.tmp_test_path / "featurevaluebarplot_js/"
        PathBuilder.remove_old_and_build(path)

        # the two groups put all their mass on different features, so the divergence is at its maximum of 1
        disjoint = [[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]]
        report = self._build_report(self._create_two_group_dataset(path, disjoint), path)
        result = report._generate()

        divergence_path = path / "jensen_shannon_divergence.csv"
        self.assertTrue(divergence_path.is_file())
        self.assertIn(divergence_path, [output.path for output in result.output_tables])

        divergences = pd.read_csv(divergence_path)
        self.assertSetEqual(set(divergences.columns),
                            {"group_1", "group_2", "jensen_shannon_divergence", "jensen_shannon_distance"})
        self.assertEqual(len(divergences), 1)
        self.assertSetEqual({divergences.loc[0, "group_1"], divergences.loc[0, "group_2"]},
                            {"original", "generated"})
        self.assertAlmostEqual(divergences.loc[0, "jensen_shannon_divergence"], 1.0)
        # the distance is the square root of the divergence
        self.assertAlmostEqual(divergences.loc[0, "jensen_shannon_distance"] ** 2,
                               divergences.loc[0, "jensen_shannon_divergence"])

        shutil.rmtree(path)

    def test_jensen_shannon_divergence_identical_groups(self):
        path = EnvironmentSettings.tmp_test_path / "featurevaluebarplot_js_identical/"
        PathBuilder.remove_old_and_build(path)

        # the same feature distribution in both groups, at different scales: the divergence is 0 either way
        identical = [[1., 2., 1.], [1., 2., 1.], [2., 4., 2.], [2., 4., 2.]]
        report = self._build_report(self._create_two_group_dataset(path, identical), path)
        report._generate()

        divergences = pd.read_csv(path / "jensen_shannon_divergence.csv")
        self.assertAlmostEqual(divergences.loc[0, "jensen_shannon_divergence"], 0.)

        shutil.rmtree(path)

    def test_jensen_shannon_divergence_against_reference(self):
        path = EnvironmentSettings.tmp_test_path / "featurevaluebarplot_js_reference/"
        PathBuilder.remove_old_and_build(path)

        encoded_data = {'examples': sparse.csr_matrix(np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])),
                        'example_ids': ["e1", "e2", "e3"],
                        'labels': {},
                        'feature_names': ["AAA", "CCC", "DDD"],
                        'encoding': "random"}
        metadata_filepath = path / "metadata.csv"
        pd.DataFrame({"subject": [1, 2, 3],
                      "origin": ["original", "model_a", "model_b"]}).to_csv(metadata_filepath, index=False)
        dataset = RepertoireDataset(encoded_data=EncodedData(**encoded_data), metadata_file=metadata_filepath)

        # with a reference, every other class is compared against it instead of all pairs being compared
        report = self._build_report(dataset, path, jensen_shannon_reference="original")
        report._generate()

        divergences = pd.read_csv(path / "jensen_shannon_divergence.csv")
        self.assertEqual(len(divergences), 2)
        self.assertListEqual(list(divergences["group_1"]), ["original", "original"])
        self.assertSetEqual(set(divergences["group_2"]), {"model_a", "model_b"})

        # a reference that is not one of the classes falls back to all pairs, with a warning
        report = self._build_report(dataset, path, jensen_shannon_reference="not_a_class")
        with self.assertLogs(level="WARNING"):
            report._generate()

        divergences = pd.read_csv(path / "jensen_shannon_divergence.csv")
        self.assertEqual(len(divergences), 3)

        shutil.rmtree(path)

    def test_jensen_shannon_divergence_skipped(self):
        path = EnvironmentSettings.tmp_test_path / "featurevaluebarplot_js_skipped/"
        PathBuilder.remove_old_and_build(path)

        disjoint = [[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]]
        dataset = self._create_two_group_dataset(path, disjoint)

        # not requested at all: no file, and the rest of the report is still produced
        report = self._build_report(dataset, path, compute_jensen_shannon=False)
        result = report._generate()
        self.assertFalse((path / "jensen_shannon_divergence.csv").is_file())
        self.assertIsInstance(result, ReportResult)

        # a negative mean feature value, as scaling produces, cannot be read as part of a distribution,
        # so the divergence is skipped with a warning while the rest of the report is still produced
        negative = [[-1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., 1., 0.]]
        report = self._build_report(self._create_two_group_dataset(path, negative), path)
        with self.assertLogs(level="WARNING"):
            result = report._generate()
        self.assertFalse((path / "jensen_shannon_divergence.csv").is_file())
        self.assertIsInstance(result, ReportResult)

        shutil.rmtree(path)
