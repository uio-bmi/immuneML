import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.hyperparameter_optimization.config.ManualSplitConfig import ManualSplitConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams
from immuneML.workflows.steps.data_splitter.ManualSplitter import ManualSplitter


class TestManualSplitter(TestCase):
    def test__split_repertoire_dataset(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "manual_splitter/")
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(10, {4: 1}, {3: 1}, {}, path)

        train_metadata = pd.DataFrame({"subject_id": ["rep_1", "rep_2", "rep_4", "rep_5", "rep_9", "rep_7"]})
        train_metadata.to_csv(path / "train.csv")

        test_metadata = pd.DataFrame({"subject_id": ["rep_0", "rep_3", "rep_6", "rep_8"]})
        test_metadata.to_csv(path / "test.csv")

        train_datasets, test_datasets = ManualSplitter._split_repertoire_dataset(
            DataSplitterParams(dataset, SplitType.MANUAL, split_count=1, paths=[path / 'result/'],
                               split_config=SplitConfig(manual_config=ManualSplitConfig(path / "train.csv",
                                                                                        path / "test.csv"),
                                                        split_count=1, split_strategy=SplitType.MANUAL)))

        self.assertEqual(1, len(train_datasets))
        self.assertEqual(1, len(test_datasets))
        self.assertEqual(6, train_datasets[0].get_example_count())
        self.assertEqual(4, test_datasets[0].get_example_count())
        self.assertTrue(all(subject_id in ["rep_1", "rep_2", "rep_4", "rep_5", "rep_9", "rep_7"]
                            for subject_id in train_datasets[0].get_metadata(["subject_id"])["subject_id"]))
        self.assertTrue(all(subject_id in ["rep_0", "rep_3", "rep_6", "rep_8"]
                            for subject_id in test_datasets[0].get_metadata(["subject_id"])["subject_id"]))
        self.assertTrue(os.path.isfile(train_datasets[0].metadata_file))
        self.assertTrue(os.path.isfile(test_datasets[0].metadata_file))

        shutil.rmtree(path)
