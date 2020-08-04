import shutil
from unittest import TestCase

from source.encodings.EncoderParams import EncoderParams
from source.encodings.atchley_kmer_encoding.AtchleyKmerEncoder import AtchleyKmerEncoder
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.util.PathBuilder import PathBuilder


class TestAtchleyKmerEncoder(TestCase):
    def test_encode(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path + "atchley_kmer_encoding/")
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(3, {1: 1}, {4: 1}, {"l1": {True: 0.4, False: 0.6}}, path + "dataset/")

        encoder = AtchleyKmerEncoder.build_object(dataset, **{"k": 2, "skip_first_n_aa": 1, "skip_last_n_aa": 1, "abundance": "RELATIVE_ABUNDANCE",
                                                              "normalize_all_features": False})
        encoded_dataset = encoder.encode(dataset, EncoderParams(path + "result/", LabelConfiguration(labels=[Label("l1")])))

        self.assertEqual((3, 3, 11), encoded_dataset.encoded_data.examples.shape)
        self.assertEqual(0., encoded_dataset.encoded_data.examples[0, 0, -1])

        shutil.rmtree(path)
