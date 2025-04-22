import os
import random
import shutil
from unittest import TestCase

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.subsampling.SubsamplingInstruction import SubsamplingInstruction


class TestSubsamplingInstruction(TestCase):
    def test_run(self):
        random.seed(1)
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "subsampling/")
        dataset = RandomDatasetGenerator.generate_receptor_dataset(200, labels={"epitope": {"A": 0.5, "B": 0.5}},
                                                                   path=path,
                                                                   chain_1_length_probabilities={3: 1},
                                                                   chain_2_length_probabilities={4: 1})
        dataset.name = "d1"

        inst = SubsamplingInstruction(dataset=dataset, subsampled_dataset_sizes=[100, 50],
                                      name="subsampling_inst")

        state = inst.run(path / "result/")

        self.assertEqual(2, len(state.subsampled_datasets))
        self.assertEqual(100, state.subsampled_datasets[0].get_example_count())
        self.assertEqual(50, state.subsampled_datasets[1].get_example_count())

        self.assertTrue(all(os.path.isfile(state.subsampled_dataset_paths[name]['airr'])
                            for name in [dataset.name for dataset in state.subsampled_datasets]))

        shutil.rmtree(path)

    def test_run_repertoire_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "subsampling_repertoire_dataset/")
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(200,
                                                                     labels={"epitope": {"A": 0.5, "B": 0.5}},
                                                                     path=path,
                                                                     sequence_count_probabilities={10: 1},
                                                                     sequence_length_probabilities={4: 1})
        dataset.name = "d1"

        inst = SubsamplingInstruction(dataset=dataset, subsampled_dataset_sizes=[100],
                                      subsampled_repertoire_size=5,
                                      name="subsampling_inst")

        state = inst.run(path / "result/")

        self.assertEqual(1, len(state.subsampled_datasets))
        self.assertEqual(100, state.subsampled_datasets[0].get_example_count())
        self.assertEqual(5, state.subsampled_datasets[0].repertoires[0].element_count)

        self.assertTrue(all(os.path.isfile(state.subsampled_dataset_paths[name]['airr'])
                            for name in [dataset.name for dataset in state.subsampled_datasets]))

        shutil.rmtree(path)
