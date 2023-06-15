import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.atchley_kmer_encoding.AtchleyKmerEncoder import AtchleyKmerEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestAtchleyKmerEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "atchley_kmer_encoding/")
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(3, {1: 1}, {4: 1}, {"l1": {True: 0.4, False: 0.6}}, path / "dataset")

        encoder = AtchleyKmerEncoder.build_object(dataset, **{"k": 2, "skip_first_n_aa": 1, "skip_last_n_aa": 1, "abundance": "RELATIVE_ABUNDANCE",
                                                              "normalize_all_features": False})
        encoded_dataset = encoder.encode(dataset, EncoderParams(path / "result", LabelConfiguration(labels=[Label("l1", [True, False])])))

        self.assertEqual((3, 11, 3), encoded_dataset.encoded_data.examples.shape)
        self.assertEqual(0., encoded_dataset.encoded_data.examples[0, -1, 0])

        shutil.rmtree(path)
