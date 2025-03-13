import os
import shutil
from unittest import TestCase

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DiversityProfile import DiversityProfile
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestDiversityProfile(TestCase):


    def test_generate_repertoire_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "diversity_profile/")

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=5,
                                                                     sequence_count_probabilities={5: 0.5, 10: 0.3, 20: 0.2},
                                                                     sequence_length_probabilities={10: 1},
                                                                     labels={"l1": {"a": 0.5, "b": 0.5}},
                                                                     path=path / "dataset")

        params = {"dataset": dataset, "result_path": path / "result"}

        report = DiversityProfile.build_object(**params)
        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        shutil.rmtree(path)
