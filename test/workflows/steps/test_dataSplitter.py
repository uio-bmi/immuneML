import os
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.data_splitter.DataSplitter import DataSplitter
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams


class TestDataSplitter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):
        PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "data_splitter/")

        dataset = RepertoireDataset(repertoires=[Repertoire(Path(f"{index}.tsv"), None, str(index))
                                                 for index in range(15)])

        paths = [EnvironmentSettings.tmp_test_path / "data_splitter/split_{}".format(i) for i in range(5)]
        for path in paths:
            PathBuilder.build(path)

        df = pd.DataFrame(data={"key1": [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1], "filename": list(range(15))})
        df.to_csv(EnvironmentSettings.tmp_test_path / "data_splitter/metadata.csv")

        dataset.metadata_file = EnvironmentSettings.tmp_test_path / "data_splitter/metadata.csv"

        training_percentage = 0.7

        trains, tests = DataSplitter.run(DataSplitterParams(
            dataset=dataset,
            training_percentage=training_percentage,
            split_strategy=SplitType.RANDOM,
            split_count=5,
            paths=paths
        ))

        self.assertTrue(isinstance(trains[0], RepertoireDataset))
        self.assertTrue(isinstance(tests[0], RepertoireDataset))
        self.assertEqual(10, len(trains[0].get_data()))
        self.assertEqual(5, len(tests[0].get_data()))
        self.assertEqual(5, len(trains))
        self.assertEqual(5, len(tests))
        self.assertEqual(10, len(trains[0].repertoires))

        paths = [EnvironmentSettings.tmp_test_path / "data_splitter/split_{}".format(i) for i in range(dataset.get_example_count())]
        for path in paths:
            PathBuilder.build(path)

        trains, tests = DataSplitter.run(DataSplitterParams(
            dataset=dataset,
            split_strategy=SplitType.LOOCV,
            split_count=-1,
            training_percentage=-1,
            paths=paths
        ))

        self.assertTrue(isinstance(trains[0], RepertoireDataset))
        self.assertTrue(isinstance(tests[0], RepertoireDataset))
        self.assertEqual(14, len(trains[0].get_data()))
        self.assertEqual(1, len(tests[0].get_data()))
        self.assertEqual(15, len(trains))
        self.assertEqual(15, len(tests))

        paths = [EnvironmentSettings.tmp_test_path / "data_splitter/split_{}".format(i) for i in range(5)]
        for path in paths:
            PathBuilder.build(path)

        trains, tests = DataSplitter.run(DataSplitterParams(
            dataset=dataset,
            split_strategy=SplitType.K_FOLD,
            split_count=5,
            training_percentage=-1,
            paths=paths
        ))

        self.assertTrue(isinstance(trains[0], RepertoireDataset))
        self.assertTrue(isinstance(tests[0], RepertoireDataset))
        self.assertEqual(len(trains[0].get_data()), 12)
        self.assertEqual(len(tests[0].get_data()), 3)
        self.assertEqual(5, len(trains))
        self.assertEqual(5, len(tests))

        trains, tests = DataSplitter.run(DataSplitterParams(
            dataset=dataset,
            split_strategy=SplitType.STRATIFIED_K_FOLD,
            split_count=3,
            training_percentage=-1,
            paths=paths,
            label_config=LabelConfiguration([Label("key1", [0, 1, 2])])
        ))

        self.assertEqual(len(trains[0].get_data()), 10)
        self.assertEqual(len(tests[0].get_data()), 5)
        self.assertEqual(3, len(trains))
        self.assertEqual(3, len(tests))
        for train in trains:
            self.assertTrue(all(cls in train.get_metadata(["key1"])["key1"] for cls in [0, 1, 2]))
        for test in tests:
            self.assertTrue(all(cls in test.get_metadata(["key1"])["key1"] for cls in [0, 1, 2]))

        shutil.rmtree(EnvironmentSettings.tmp_test_path / "data_splitter/")
