import shutil
from unittest import TestCase

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator


class TestRandomDatasetGenerator(TestCase):
    def test_generate_repertoire_dataset(self):

        path = f"{EnvironmentSettings.tmp_test_path}random_dataset_generation/"

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=100,
                                                                     sequence_count_probabilities={5: 0.5, 6: 0.5},
                                                                     sequence_length_probabilities={4: 0.5, 5: 0.5},
                                                                     labels={"HLA": {"A": 0.5, "B": 0.5}},
                                                                     path=path)

        self.assertEqual(100, dataset.get_example_count())
        for repertoire in dataset.repertoires:
            self.assertTrue(repertoire.get_element_count() == 5 or repertoire.get_element_count() == 6)
            self.assertTrue(all(len(sequence_aa) in [4, 5] for sequence_aa in repertoire.get_sequence_aas().tolist()))
            self.assertTrue(repertoire.metadata["HLA"] in ["A", "B"])

        shutil.rmtree(path)
