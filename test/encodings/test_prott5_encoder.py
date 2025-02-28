import os
import shutil

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.prott5.ProtT5Encoder import ProtT5Encoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestProtT5Encoder:

    def setup_method(self):
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name
        self.base_path = EnvironmentSettings.tmp_test_path / "prot_t5_encoder/"
        PathBuilder.remove_old_and_build(self.base_path)

    def teardown_method(self):
        shutil.rmtree(self.base_path)

    def create_encoder(self, device='cpu'):
        return ProtT5Encoder(
            device=device
        )

    def create_label_config(self):
        return LabelConfiguration([Label("label", [True, False])])

    def test_encode_sequence_dataset(self):
        path = self.base_path / "sequence/"
        PathBuilder.remove_old_and_build(path)

        encoder = self.create_encoder()
        dataset = RandomDatasetGenerator.generate_sequence_dataset(sequence_count=10, length_probabilities={3: 1},
                                                                   labels={"label": {True: 0.5, False: 0.5}}, path=path)

        lc = self.create_label_config()
        encoded = encoder.encode(dataset, EncoderParams(
            label_config=lc,
            pool_size=4,
            learn_model=True
        ))

        assert encoded.encoded_data is not None
        assert encoded.encoded_data.examples.shape[0] == 10  # number of sequences
        assert encoded.encoded_data.examples.shape[1] == 1024  # embedding dimension
        assert len(encoded.encoded_data.example_ids) == 10
        assert all(isinstance(label, bool) for label in encoded.encoded_data.labels["label"])

    def test_encode_receptor_dataset(self):
        path = self.base_path / "receptor/"
        PathBuilder.remove_old_and_build(path)

        encoder = self.create_encoder()
        dataset = RandomDatasetGenerator.generate_receptor_dataset(receptor_count=10,
                                                                   chain_1_length_probabilities={3: 1},
                                                                   chain_2_length_probabilities={3: 1},
                                                                   labels={"label": {True: 0.5, False: 0.5}}, path=path)

        lc = self.create_label_config()
        encoded = encoder.encode(dataset, EncoderParams(
            label_config=lc,
            pool_size=4,
            learn_model=True
        ))

        assert encoded.encoded_data is not None
        assert encoded.encoded_data.examples.shape[0] == 10  # number of receptors
        assert encoded.encoded_data.examples.shape[1] == 2048  # embedding dimension (2 * 1024)
        assert isinstance(encoded.encoded_data.examples, np.ndarray)
        assert encoded.encoded_data.examples.dtype == np.float32
        assert len(encoded.encoded_data.example_ids) == 10
        assert all(isinstance(label, bool) for label in encoded.encoded_data.labels["label"])

    def test_encode_repertoire_dataset(self):
        path = self.base_path / "repertoire/"
        PathBuilder.remove_old_and_build(path)

        encoder = self.create_encoder()
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=10,
                                                                     sequence_count_probabilities={10: 1},
                                                                     sequence_length_probabilities={3: 1},
                                                                     labels={"label": {True: 0.5, False: 0.5}},
                                                                     path=path)

        lc = self.create_label_config()
        encoded = encoder.encode(dataset, EncoderParams(
            label_config=lc,
            pool_size=4,
            learn_model=True
        ))

        assert encoded.encoded_data is not None
        assert encoded.encoded_data.examples.shape[0] == 10  # number of repertoires
        assert len(encoded.encoded_data.example_ids) == 10
        assert all(isinstance(label, bool) for label in encoded.encoded_data.labels["label"])
