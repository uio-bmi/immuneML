import shutil
from unittest import TestCase

from immuneML.data_model.SequenceParams import ChainPair
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.data_model.SequenceSet import Receptor
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.hyperparameter_optimization.config.LeaveOneOutConfig import LeaveOneOutConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams
from immuneML.workflows.steps.data_splitter.LeaveOneOutSplitter import LeaveOneOutSplitter


class TestLeaveOneOutSplitter(TestCase):
    def test_split_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "leave_one_out_splitter/")
        receptors = []
        for i in range(10):
            receptors.append(Receptor(chain_1=ReceptorSequence(locus='alpha'),
                                          chain_2=ReceptorSequence(locus='beta'), chain_pair=ChainPair.TRA_TRB,
                                          metadata={"subject": i % 3}, receptor_id=str(i), cell_id=str(i)))

        dataset = ReceptorDataset.build_from_objects(receptors, path, 'd1')

        params = DataSplitterParams(dataset, SplitType.LEAVE_ONE_OUT_STRATIFICATION, 3, paths=[path / f"result_{i}/" for i in range(1, 4)],
                                    split_config=SplitConfig(SplitType.LEAVE_ONE_OUT_STRATIFICATION, split_count=3,
                                                             leave_one_out_config=LeaveOneOutConfig("subject", 1)))
        train_datasets, test_datasets = LeaveOneOutSplitter.split_dataset(params)

        self.assertEqual(3, len(train_datasets))
        self.assertEqual(3, len(test_datasets))

        for i in range(3):
            self.assertTrue(all(receptor.metadata["subject"] == i for receptor in test_datasets[i].get_data()))
            self.assertTrue(all(receptor.metadata["subject"] != i for receptor in train_datasets[i].get_data()))

        shutil.rmtree(path)

    def test_split_rep_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "leave_one_out_splitter_rep/")
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(30, {1: 1.}, {1: 1.},
                                                                     {'batch': {0: 0.33, 1: 0.33, 2: 0.33}},
                                                                     path, 'd1')

        params = DataSplitterParams(dataset, SplitType.LEAVE_ONE_OUT_STRATIFICATION, 3,
                                    paths=[path / f"result_{i}/" for i in range(1, 4)],
                                    split_config=SplitConfig(SplitType.LEAVE_ONE_OUT_STRATIFICATION, split_count=3,
                                                             leave_one_out_config=LeaveOneOutConfig("batch", 1)))
        train_datasets, test_datasets = LeaveOneOutSplitter.split_dataset(params)

        self.assertEqual(3, len(train_datasets))
        self.assertEqual(3, len(test_datasets))

        for i in range(3):
            self.assertTrue(all(repertoire.metadata["batch"] == i for repertoire in test_datasets[i].get_data()))
            self.assertTrue(all(repertoire.metadata["batch"] != i for repertoire in train_datasets[i].get_data()))

        shutil.rmtree(path)

