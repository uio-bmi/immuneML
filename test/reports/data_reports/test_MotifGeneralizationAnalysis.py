import os
import shutil
from unittest import TestCase

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.MotifGeneralizationAnalysis import MotifGeneralizationAnalysis
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestMotifGeneralizationAnalysis(TestCase):
    def test_generate(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "significant_motif_overlap/")


        dataset = RandomDatasetGenerator.generate_sequence_dataset(100, {10: 1}, {"l1": {"A": 0.5, "B": 0.5}}, path / "dataset")

        # sequence_count: int, length_probabilities: dict, labels: dict, path: Path

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "MotifGeneralizationAnalysis")
        params["max_positions"] = 1
        params["min_precision"] = 0.8
        params["random_seed"] = 1
        params["dataset"] = dataset
        params["result_path"] = path / "result"
        params["label"] = {"l1": {"positive_class": "A"}}

        report = MotifGeneralizationAnalysis.build_object(**params)

        report._generate()


        self.assertTrue(os.path.isdir(path / "result/datasets/train"))
        self.assertTrue(os.path.isdir(path / "result/datasets/test"))
        self.assertTrue(os.path.isdir(path / "result/encoded_data"))

        self.assertTrue(os.path.isfile(path / "result/training_set_scores.csv"))
        self.assertTrue(os.path.isfile(path / "result/test_set_scores.csv"))

        self.assertTrue(os.path.isfile(path / "result/train_precision_per_tp.html"))
        self.assertTrue(os.path.isfile(path / "result/test_precision_per_tp.html"))

        self.assertTrue(os.path.isfile(path / "result/train_precision_recall.html"))
        self.assertTrue(os.path.isfile(path / "result/test_precision_recall.html"))

        shutil.rmtree(path)
