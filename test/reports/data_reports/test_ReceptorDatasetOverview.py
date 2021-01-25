import os
import shutil
from unittest import TestCase

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.ReceptorDatasetOverview import ReceptorDatasetOverview
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestReceptorDatasetOverview(TestCase):
    def test_generate(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "receptor_dataset_overview/")

        dataset = RandomDatasetGenerator.generate_receptor_dataset(100, {9: 0.3, 10: 0.4, 11: 0.1, 12: 0.2}, {9: 0.1, 10: 0.2, 11: 0.4, 12: 0.3},
                                                                   {}, path / "dataset")

        report = ReceptorDatasetOverview(200, dataset, path / "result", "receptor_overview")
        result = report.generate_report()

        self.assertTrue(os.path.isfile(path / "result/sequence_length_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/sequence_length_distribution_chain_alpha.csv"))
        self.assertTrue(os.path.isfile(path / "result/sequence_length_distribution_chain_beta.csv"))
        self.assertTrue(isinstance(result, ReportResult))

        shutil.rmtree(path)
