import os
import random
import shutil
from unittest import TestCase

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
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

    def test_run_class_balanced(self):
        random.seed(2)
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "subsampling_class_balanced/")
        dataset = RandomDatasetGenerator.generate_receptor_dataset(300, labels={"epitope": {"A": 0.2, "B": 0.8}},
                                                                   path=path,
                                                                   chain_1_length_probabilities={3: 1},
                                                                   chain_2_length_probabilities={4: 1})
        dataset.name = "d1"

        label = Label(name="epitope", values=["A", "B"])

        inst = SubsamplingInstruction(dataset=dataset, subsampled_dataset_sizes=[50, 50],
                                      label=label,
                                      subsampled_class_distributions=[{"A": 0.1, "B": 0.9}, {"A": 0.5, "B": 0.5}],
                                      name="subsampling_inst")

        state = inst.run(path / "result/")

        self.assertEqual(2, len(state.subsampled_datasets))

        counts_0 = state.subsampled_datasets[0].get_metadata(["epitope"], return_df=True)["epitope"].value_counts()
        self.assertEqual(50, state.subsampled_datasets[0].get_example_count())
        self.assertEqual(5, counts_0["A"])
        self.assertEqual(45, counts_0["B"])

        counts_1 = state.subsampled_datasets[1].get_metadata(["epitope"], return_df=True)["epitope"].value_counts()
        self.assertEqual(50, state.subsampled_datasets[1].get_example_count())
        self.assertEqual(25, counts_1["A"])
        self.assertEqual(25, counts_1["B"])

        self.assertTrue(all(os.path.isfile(state.subsampled_dataset_paths[name]['airr'])
                            for name in [dataset.name for dataset in state.subsampled_datasets]))

        shutil.rmtree(path)

    def test_run_class_balanced_infeasible(self):
        random.seed(3)
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "subsampling_class_balanced_infeasible/")
        dataset = RandomDatasetGenerator.generate_receptor_dataset(50, labels={"epitope": {"A": 0.1, "B": 0.9}},
                                                                   path=path,
                                                                   chain_1_length_probabilities={3: 1},
                                                                   chain_2_length_probabilities={4: 1})
        dataset.name = "d1"

        n_a = int(dataset.get_metadata(["epitope"], return_df=True)["epitope"].value_counts().get("A", 0))
        requested_size = n_a + 10  # always more 'A' examples than exist in the dataset

        label = Label(name="epitope", values=["A", "B"])

        inst = SubsamplingInstruction(dataset=dataset, subsampled_dataset_sizes=[requested_size],
                                      label=label,
                                      subsampled_class_distributions=[{"A": 1.0, "B": 0.0}],
                                      name="subsampling_inst")

        self.assertRaises(AssertionError, inst.run, path / "result/")

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
        self.assertEqual(5, state.subsampled_datasets[0].repertoires[0].get_element_count())

        self.assertTrue(all(os.path.isfile(state.subsampled_dataset_paths[name]['airr'])
                            for name in [dataset.name for dataset in state.subsampled_datasets]))

        shutil.rmtree(path)
