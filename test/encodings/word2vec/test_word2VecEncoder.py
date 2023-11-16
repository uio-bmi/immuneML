import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.word2vec.W2VRepertoireEncoder import W2VRepertoireEncoder, Word2VecEncoder
from immuneML.encodings.word2vec.W2VSequenceEncoder import W2VSequenceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestWord2VecEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode_repertoire(self):

        test_path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "w2v_repertoire/")

        sequence1 = ReceptorSequence("CASSVFA", sequence_id="1")
        sequence2 = ReceptorSequence("CASSCCC", sequence_id="2")

        metadata1 = {"T1D": "T1D", "subject_id": "1"}
        rep1 = Repertoire.build_from_sequence_objects([sequence1, sequence2], test_path, metadata1)

        metadata2 = {"T1D": "CTL", "subject_id": "2"}
        rep2 = Repertoire.build_from_sequence_objects([sequence1], test_path, metadata2)

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        label_configuration = LabelConfiguration()
        label_configuration.add_label("T1D", ["T1D", "CTL"])

        config_params = EncoderParams(
            model={},
            learn_model=True,
            result_path=test_path,
            label_config=label_configuration,
        )

        encoder = Word2VecEncoder.build_object(dataset, **{
                "k": 3,
                "model_type": "sequence",
                "vector_size": 16,
                "epochs": 10,
                "window": 5
            })

        encoded_dataset = encoder.encode(dataset=dataset, params=config_params)

        self.assertIsNotNone(encoded_dataset.encoded_data)
        self.assertTrue(encoded_dataset.encoded_data.examples.shape[0] == 2)
        self.assertTrue(encoded_dataset.encoded_data.examples.shape[1] == 16)
        self.assertTrue(len(encoded_dataset.encoded_data.labels["T1D"]) == 2)
        self.assertTrue(encoded_dataset.encoded_data.labels["T1D"][0] == "T1D")
        self.assertTrue(isinstance(encoder, W2VRepertoireEncoder))

        shutil.rmtree(test_path)

    def test_encode_sequences(self):

        test_path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "w2v_seqs/")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {6: 1.}, {"l1": {True: 0.5, False: 0.5}}, path=test_path)

        label_configuration = LabelConfiguration()
        label_configuration.add_label("l1", [True, False])

        config_params = EncoderParams(
            model={},
            learn_model=True,
            result_path=test_path / 'encoded',
            label_config=label_configuration,
        )

        encoder = Word2VecEncoder.build_object(dataset, **{
                "k": 3,
                "model_type": "sequence",
                "vector_size": 16,
                "epochs": 10,
                "window": 5
            })

        encoded_dataset = encoder.encode(dataset=dataset, params=config_params)

        self.assertIsNotNone(encoded_dataset.encoded_data)
        self.assertTrue(encoded_dataset.encoded_data.examples.shape[0] == 10)
        self.assertTrue(encoded_dataset.encoded_data.examples.shape[1] == 16)
        self.assertTrue(isinstance(encoder, W2VSequenceEncoder))

        shutil.rmtree(test_path)
