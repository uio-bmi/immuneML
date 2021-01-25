import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.data_splitter.DataSplitter import DataSplitter
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams


class TestDataSplitter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):
        dataset = RepertoireDataset(repertoires=[Repertoire("0.npy", None, "0"),
                                                 Repertoire("0.npy", None, "1"),
                                                 Repertoire("0.npy", None, "2"),
                                                 Repertoire("0.npy", None, "3"),
                                                 Repertoire("0.npy", None, "4"),
                                                 Repertoire("0.npy", None, "5"),
                                                 Repertoire("0.npy", None, "6"),
                                                 Repertoire("0.npy", None, "7")])

        paths = [EnvironmentSettings.root_path / "test/tmp/datasplitter/split_{}".format(i) for i in range(5)]
        for path in paths:
            PathBuilder.build(path)

        df = pd.DataFrame(data={"key1": [0, 0, 1, 1, 1, 2, 2, 0], "filename": [0, 1, 2, 3, 4, 5, 6, 7]})
        df.to_csv(EnvironmentSettings.root_path / "test/tmp/datasplitter/metadata.csv")

        dataset.metadata_file = EnvironmentSettings.root_path / "test/tmp/datasplitter/metadata.csv"

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
        self.assertEqual(len(trains[0].get_data()), 5)
        self.assertEqual(len(tests[0].get_data()), 3)
        self.assertEqual(5, len(trains))
        self.assertEqual(5, len(tests))
        self.assertEqual(5, len(trains[0].repertoires))

        trains2, tests2 = DataSplitter.run(DataSplitterParams(
            dataset=dataset,
            training_percentage=training_percentage,
            split_strategy=SplitType.RANDOM,
            split_count=5,
            paths=paths
        ))

        self.assertEqual(trains[0].get_repertoire_ids(), trains2[0].get_repertoire_ids())

        paths = [EnvironmentSettings.root_path / "test/tmp/datasplitter/split_{}".format(i) for i in range(dataset.get_example_count())]
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
        self.assertEqual(len(trains[0].get_data()), 7)
        self.assertEqual(len(tests[0].get_data()), 1)
        self.assertEqual(8, len(trains))
        self.assertEqual(8, len(tests))

        paths = [EnvironmentSettings.root_path / "test/tmp/datasplitter/split_{}".format(i) for i in range(5)]
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
        self.assertEqual(len(trains[0].get_data()), 6)
        self.assertEqual(len(tests[0].get_data()), 2)
        self.assertEqual(5, len(trains))
        self.assertEqual(5, len(tests))

        shutil.rmtree(EnvironmentSettings.root_path / "test/tmp/datasplitter/")
