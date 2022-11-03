import os
import shutil
from unittest import TestCase

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.MotifOverlap import MotifOverlap
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestMotifOverlap(TestCase):
    def test_generate(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "significant_motif_overlap/")


        dataset = RandomDatasetGenerator.generate_sequence_dataset(100, {10: 1}, {"l1": {"A": 0.5, "B": 0.5}}, path / "dataset")

        # sequence_count: int, length_probabilities: dict, labels: dict, path: Path

        report = MotifOverlap.build_object(**{"n_splits": 5,
                                                      "max_positions": 1,
                                                      "min_precision": 0.8,
                                                      "min_recall": 0.01,
                                                      "min_true_positives": 1,
                                                      "dataset": dataset,
                                                      "random_seed": 1,
                                              "result_path": path / "result",
                                              "label": {"l1": {"positive_class": "A"}}})

        report._generate()


        self.assertTrue(os.path.isdir(path / "result/datasets"))
        self.assertTrue(os.path.isdir(path / "result/encoded_data"))
        self.assertTrue(os.path.isdir(path / "result/feature_intersections"))

        for i in range(5):
            self.assertTrue(os.path.isdir(path / f"result/datasets/split_{i}"))
            self.assertTrue(os.path.isdir(path / f"result/encoded_data/split_{i}"))

        self.assertTrue(os.path.isfile(path / "result/feature_intersections/multi_intersection_motifs.tsv"))
        self.assertTrue(os.path.isfile(path / "result/feature_intersections/number_of_features_per_subset.tsv"))
        self.assertTrue(os.path.isfile(path / "result/feature_intersections/pairwise_intersections.tsv"))

        shutil.rmtree(path)
