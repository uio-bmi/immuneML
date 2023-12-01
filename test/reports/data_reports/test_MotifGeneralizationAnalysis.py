import os
import shutil
import pandas as pd
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

        identifiers = [seq.identifier for seq in dataset.get_data()]
        training_set_identifiers = identifiers[::2]

        with open(path / "training_ids.txt", "w") as identifiers_file:
            identifiers_file.writelines("example_id\n")
            identifiers_file.writelines([identifier + "\n" for identifier in training_set_identifiers])

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "MotifGeneralizationAnalysis")
        params["training_set_identifier_path"] = str(path / "training_ids.txt")
        params["min_positions"] = 1
        params["max_positions"] = 1
        params["min_precision"] = 0.8
        params["split_by_motif_size"] = True
        params["random_seed"] = 1
        params["dataset"] = dataset
        params["result_path"] = path / "result"
        params["label"] = {"l1": {"positive_class": "A"}}

        report = MotifGeneralizationAnalysis.build_object(**params)

        report._generate()


        self.assertTrue(os.path.isdir(path / "result/datasets/train"))
        self.assertTrue(os.path.isdir(path / "result/datasets/test"))
        self.assertTrue(os.path.isdir(path / "result/encoded_data"))

        self.assertTrue(os.path.isfile(path / "result/training_set_scores_motif_size=1.csv"))
        self.assertTrue(os.path.isfile(path / "result/test_set_scores_motif_size=1.csv"))
        self.assertTrue(os.path.isfile(path / "result/training_combined_precision_motif_size=1.csv"))
        self.assertTrue(os.path.isfile(path / "result/test_combined_precision_motif_size=1.csv"))

        self.assertTrue(os.path.isfile(path / "result/training_precision_per_tp_motif_size=1.html"))
        self.assertTrue(os.path.isfile(path / "result/test_precision_per_tp_motif_size=1.html"))

        self.assertTrue(os.path.isfile(path / "result/training_precision_recall_motif_size=1.html"))
        self.assertTrue(os.path.isfile(path / "result/test_precision_recall_motif_size=1.html"))

        self.assertTrue(os.path.isfile(path / "result/tp_recall_cutoffs.tsv"))

        shutil.rmtree(path)


    def test_set_tp_cutoff(self):
        test_df = pd.DataFrame({"training_TP": [1, 2, 3, 4, 5, 6, 7, 8], "combined_precision": [0.1, 0.2, 0.3, 0.4, 0.8, 0.6, 0.7, 0.8]})
        ma = MotifGeneralizationAnalysis()

        ma.test_precision_threshold = 0.7
        self.assertEqual(ma._determine_tp_cutoff(test_df), 7)

        ma.test_precision_threshold = 1
        self.assertEqual(ma._determine_tp_cutoff(test_df), None)

