import shutil
from unittest import TestCase

from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator


class TestRandomDatasetGenerator(TestCase):
    def test_generate_repertoire_dataset(self):

        path = EnvironmentSettings.tmp_test_path / "random_repertoire_dataset_generation/"

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=100,
                                                                     sequence_count_probabilities={5: 0.5, 6: 0.5},
                                                                     sequence_length_probabilities={4: 0.5, 5: 0.5},
                                                                     labels={"HLA": {"A": 0.5, "B": 0.5}},
                                                                     path=path)

        self.assertEqual(RepertoireDataset, type(dataset))

        self.assertEqual(100, dataset.get_example_count())
        for repertoire in dataset.repertoires:
            self.assertTrue(repertoire.get_element_count() == 5 or repertoire.get_element_count() == 6)
            self.assertTrue(all(seq_len in [4, 5] for seq_len in repertoire.data.cdr3_aa.lengths.tolist()))
            self.assertTrue(repertoire.metadata["HLA"] in ["A", "B"])

        shutil.rmtree(path)

    def test_generate_receptor_dataset(self):

        path = EnvironmentSettings.tmp_test_path / "random_receptor_dataset_generation/"

        dataset = RandomDatasetGenerator.generate_receptor_dataset(receptor_count=100,
                                                                   chain_1_length_probabilities={4: 0.5, 5: 0.5},
                                                                   chain_2_length_probabilities={4: 0.5, 5: 0.5},
                                                                   labels={"HLA": {"A": 0.5, "B": 0.5}},
                                                                   path=path)

        self.assertEqual(ReceptorDataset, type(dataset))

        self.assertEqual(100, dataset.get_example_count())
        for receptor in dataset.get_data():
            self.assertTrue(len(sequence_aa) in [4, 5] for sequence_aa in [receptor.alpha.sequence_aa, receptor.beta.sequence_aa])
            self.assertTrue(receptor.metadata["HLA"] in ["A", "B"])

        shutil.rmtree(path)

    def test_generate_sequence_dataset(self):

        path = EnvironmentSettings.tmp_test_path / "random_sequence_dataset_generation/"

        dataset = RandomDatasetGenerator.generate_sequence_dataset(sequence_count=100,
                                                                   length_probabilities={4: 0.5, 5: 0.5},
                                                                   labels={"HLA": {"A": 0.5, "B": 0.5}},
                                                                   path=path)

        self.assertEqual(SequenceDataset, type(dataset))
        self.assertEqual(100, dataset.get_example_count())

        for sequence in dataset.get_data():
            self.assertTrue(len(sequence.sequence_aa) in [4, 5])
            self.assertTrue(sequence.metadata["HLA"] in ["A", "B"])

        shutil.rmtree(path)
