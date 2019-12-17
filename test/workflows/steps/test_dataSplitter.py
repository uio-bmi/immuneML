import shutil
from unittest import TestCase

import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.hyperparameter_optimization.SplitType import SplitType
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.DataSplitter import DataSplitter
from source.workflows.steps.DataSplitterParams import DataSplitterParams


class TestDataSplitter(TestCase):

    def test_run(self):
        dataset = RepertoireDataset(repertoires=[SequenceRepertoire("0.npy", "", "0"),
                                                 SequenceRepertoire("0.npy", "", "1"),
                                                 SequenceRepertoire("0.npy", "", "2"),
                                                 SequenceRepertoire("0.npy", "", "3"),
                                                 SequenceRepertoire("0.npy", "", "4"),
                                                 SequenceRepertoire("0.npy", "", "5"),
                                                 SequenceRepertoire("0.npy", "", "6"),
                                                 SequenceRepertoire("0.npy", "", "7")])

        path = EnvironmentSettings.root_path + "test/tmp/datasplitter/"
        PathBuilder.build(path)

        df = pd.DataFrame(data={"key1": [0, 0, 1, 1, 1, 2, 2, 0], "filename": [0, 1, 2, 3, 4, 5, 6, 7]})
        df.to_csv(path+"metadata.csv")

        dataset.metadata_file = path+"metadata.csv"

        training_percentage = 0.7

        trains, tests = DataSplitter.run(DataSplitterParams(
            dataset=dataset,
            training_percentage=training_percentage,
            split_strategy=SplitType.RANDOM,
            split_count=5,
            label_to_balance=None,
            path=path
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
            label_to_balance=None,
            path=path
        ))

        self.assertEqual(trains[0].get_repertoire_ids(), trains2[0].get_repertoire_ids())

        trains, tests = DataSplitter.run(DataSplitterParams(
            dataset=dataset,
            split_strategy=SplitType.LOOCV,
            split_count=-1,
            label_to_balance=None,
            training_percentage=-1,
            path=path
        ))

        self.assertTrue(isinstance(trains[0], RepertoireDataset))
        self.assertTrue(isinstance(tests[0], RepertoireDataset))
        self.assertEqual(len(trains[0].get_data()), 7)
        self.assertEqual(len(tests[0].get_data()), 1)
        self.assertEqual(8, len(trains))
        self.assertEqual(8, len(tests))

        trains, tests = DataSplitter.run(DataSplitterParams(
            dataset=dataset,
            split_strategy=SplitType.K_FOLD,
            split_count=5,
            label_to_balance=None,
            training_percentage=-1,
            path=path
        ))

        self.assertTrue(isinstance(trains[0], RepertoireDataset))
        self.assertTrue(isinstance(tests[0], RepertoireDataset))
        self.assertEqual(len(trains[0].get_data()), 6)
        self.assertEqual(len(tests[0].get_data()), 2)
        self.assertEqual(5, len(trains))
        self.assertEqual(5, len(tests))

        trains, tests = DataSplitter.run(DataSplitterParams(
            dataset=dataset,
            split_strategy=SplitType.RANDOM_BALANCED,
            training_percentage=training_percentage,
            split_count=10,
            label_to_balance="key1",
            path=path
        ))

        self.assertTrue(isinstance(trains[0], RepertoireDataset))
        self.assertTrue(isinstance(tests[0], RepertoireDataset))
        self.assertEqual(10, len(trains))
        self.assertEqual(10, len(tests))
        self.assertEqual(len(trains[0].get_data()) + len(tests[0].get_data()), 6)

        shutil.rmtree(path)
