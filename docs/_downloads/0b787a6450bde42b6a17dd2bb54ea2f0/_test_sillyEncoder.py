import os
import shutil
import unittest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.EncodedData import EncodedData
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.encodings.silly.SillyEncoder import SillyEncoder
from immuneML.environment.Label import Label
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator


class TestSillyEncoder(unittest.TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _get_mock_sequence_dataset(self, path):
        # Create a mock SequenceDataset with 10 sequences of length 15,
        # and a label called 'binding' with 50% chance of having status 'yes' or 'no'
        dataset = RandomDatasetGenerator.generate_sequence_dataset(sequence_count=10,
                                                                   length_probabilities={15: 1},
                                                                   labels={"binding": {"yes": 0.5, "no": 0.5}},
                                                                   path=path)

        label_config = LabelConfiguration(labels=[Label(name="binding", values=["yes", "no"])])

        return dataset, label_config

    def _get_mock_receptor_dataset(self, path):
        # Create a mock ReceptorDataset with 10 receptors with sequences of length 15,
        # and a label called 'binding' with 50% chance of having status 'yes' or 'no'
        dataset = RandomDatasetGenerator.generate_receptor_dataset(receptor_count=10,
                                                                   chain_1_length_probabilities={15: 1},
                                                                   chain_2_length_probabilities={15: 1},
                                                                   labels={"binding": {"yes": 0.5, "no": 0.5}},
                                                                   path=path)

        label_config = LabelConfiguration(labels=[Label(name="binding", values=["yes", "no"])])

        return dataset, label_config

    def _get_mock_repertoire_dataset(self, path):
        # Create a mock RepertoireDataset with 10 repertoires, each containing 50 sequences of length 15,
        # and a label called 'disease' with 50% chance of having status 'yes' or 'no'
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=10,
                                                           sequence_count_probabilities={50: 1},
                                                           sequence_length_probabilities={15: 1},
                                                           labels={"disease": {"yes": 0.5, "no": 0.5}},
                                                           path=path)

        label_config = LabelConfiguration(labels=[Label(name="disease", values=["yes", "no"])])

        return dataset, label_config

    def test_silly_sequence_encoder(self):
        tmp_path = EnvironmentSettings.tmp_test_path / "silly_sequence/"
        sequence_dataset, label_config = self._get_mock_sequence_dataset(tmp_path)
        self._test_silly_encoder(tmp_path, sequence_dataset, label_config)

    def test_silly_receptor_encoder(self):
        tmp_path = EnvironmentSettings.tmp_test_path / "silly_receptor/"
        receptor_dataset, label_config = self._get_mock_receptor_dataset(tmp_path)
        self._test_silly_encoder(tmp_path, receptor_dataset, label_config)

    def test_silly_repertoire_encoder(self):
        tmp_path = EnvironmentSettings.tmp_test_path / "silly_repertoire/"
        receptor_dataset, label_config = self._get_mock_repertoire_dataset(tmp_path)
        self._test_silly_encoder(tmp_path, receptor_dataset, label_config)

    def _test_silly_encoder(self, tmp_path, dataset, label_config):
        # test getting a SillyEncoder from the build_object method
        params = {"random_seed": 1, "embedding_len": 3}
        encoder = SillyEncoder.build_object(dataset, **params)
        self.assertIsInstance(encoder, SillyEncoder)

        # test encoding data
        encoded_dataset = encoder.encode(dataset,
                                         params=EncoderParams(result_path=tmp_path,
                                                              label_config=label_config))

        # the result must be a Dataset (of the same subtype as the original dataset) with EncodedData attached
        self.assertIsInstance(encoded_dataset, dataset.__class__)
        self.assertIsInstance(encoded_dataset.encoded_data, EncodedData)

        # testing the validity of the encoded data
        self.assertEqual(dataset.get_example_ids(), encoded_dataset.encoded_data.example_ids)
        self.assertTrue((encoded_dataset.encoded_data.examples >= 0).all())
        self.assertTrue((encoded_dataset.encoded_data.examples <= 1).all())

        # don't forget to remove the temporary data
        shutil.rmtree(tmp_path)
