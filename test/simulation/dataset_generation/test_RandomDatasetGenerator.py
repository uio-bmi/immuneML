import shutil
from unittest import TestCase

from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator


class TestRandomDatasetGenerator(TestCase):
    def test_generate_repertoire_dataset(self):

        path = f"{EnvironmentSettings.tmp_test_path}random_repertoire_dataset_generation/"

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=100,
                                                                     sequence_count_probabilities={5: 0.5, 6: 0.5},
                                                                     sequence_length_probabilities={4: 0.5, 5: 0.5},
                                                                     labels={"HLA": {"A": 0.5, "B": 0.5}},
                                                                     path=path)

        self.assertEqual(RepertoireDataset, type(dataset))

        self.assertEqual(100, dataset.get_example_count())
        for repertoire in dataset.repertoires:
            self.assertTrue(repertoire.get_element_count() == 5 or repertoire.get_element_count() == 6)
            self.assertTrue(all(len(sequence_aa) in [4, 5] for sequence_aa in repertoire.get_sequence_aas().tolist()))
            self.assertTrue(repertoire.metadata["HLA"] in ["A", "B"])

        shutil.rmtree(path)

    def test_generate_receptor_dataset(self):

        path = f"{EnvironmentSettings.tmp_test_path}random_receptor_dataset_generation/"

        dataset = RandomDatasetGenerator.generate_receptor_dataset(receptor_count=100,
                                                                   chain_1_length_probabilities={4: 0.5, 5: 0.5},
                                                                   chain_2_length_probabilities={4: 0.5, 5: 0.5},
                                                                   labels={"HLA": {"A": 0.5, "B": 0.5}},
                                                                   path=path)

        self.assertEqual(ReceptorDataset, type(dataset))

        self.assertEqual(100, dataset.get_example_count())
        for receptor in dataset.get_data():
            self.assertTrue(len(sequence_aa) in [4, 5] for sequence_aa in [receptor.alpha.amino_acid_sequence, receptor.beta.amino_acid_sequence])
            self.assertTrue(receptor.metadata["HLA"] in ["A", "B"])

        shutil.rmtree(path)

    def test_generate_sequence_dataset(self):

        path = f"{EnvironmentSettings.tmp_test_path}random_sequence_dataset_generation/"

        dataset = RandomDatasetGenerator.generate_sequence_dataset(sequence_count=100,
                                                                   length_probabilities={4: 0.5, 5: 0.5},
                                                                   labels={"HLA": {"A": 0.5, "B": 0.5}},
                                                                   path=path)

        self.assertEqual(SequenceDataset, type(dataset))
        self.assertEqual(100, dataset.get_example_count())

        for sequence in dataset.get_data():
            self.assertTrue(len(sequence.amino_acid_sequence) in [4, 5])
            self.assertTrue(sequence.get_attribute("HLA") in ["A", "B"])

        shutil.rmtree(path)
