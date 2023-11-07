import os
import shutil
from unittest import TestCase

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.SimpleDatasetOverview import SimpleDatasetOverview
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestSimpleDatasetOverview(TestCase):
    def test_generate_sequence_dataset(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "overview_sequence_dataset/")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(100, {10: 0.5, 11:0.25, 20:0.25}, {"l1": {"a": 0.5, "b": 0.5}}, path / "dataset")

        params = {"dataset": dataset, "result_path": path / "result"}

        report = SimpleDatasetOverview.build_object(**params)
        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertTrue(os.path.isfile(path / "result/dataset_description.txt"))
        self.assertTrue(os.path.isfile(path / "result/label_results_table.html"))

        shutil.rmtree(path)

    def test_generate_receptor_dataset(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "overview_receptor_dataset/")


        dataset = RandomDatasetGenerator.generate_receptor_dataset(100, chain_1_length_probabilities={10: 0.5, 11:0.25, 20:0.25},
                                                                   chain_2_length_probabilities={10: 0.5, 11: 0.25, 15: 0.25},
                                                                   labels={"l1": {"a": 0.5, "b": 0.5}}, path=path / "dataset")

        params = {"dataset": dataset, "result_path": path / "result"}

        report = SimpleDatasetOverview.build_object(**params)
        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertTrue(os.path.isfile(path / "result/dataset_description.txt"))

        shutil.rmtree(path)

    def test_generate_repertoire_dataset(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "overview_repertoire_dataset/")


        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=5, sequence_count_probabilities={20:1},
                                                                     sequence_length_probabilities={10: 1},
                                                                     labels={"l1": {"a": 0.5, "b": 0.5}}, path=path / "dataset")

        params = {"dataset": dataset, "result_path": path / "result"}

        report = SimpleDatasetOverview.build_object(**params)
        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)
        self.assertTrue(os.path.isfile(path / "result/dataset_description.txt"))

        shutil.rmtree(path)