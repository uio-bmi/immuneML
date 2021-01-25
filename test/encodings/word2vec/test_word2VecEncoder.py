import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.word2vec.W2VRepertoireEncoder import W2VRepertoireEncoder, Word2VecEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestWord2VecEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):

        test_path = EnvironmentSettings.root_path / "test/tmp/w2v/"

        PathBuilder.build(test_path)

        sequence1 = ReceptorSequence("CASSVFA", identifier="1")
        sequence2 = ReceptorSequence("CASSCCC", identifier="2")

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
            filename="dataset.pkl"
        )

        encoder = Word2VecEncoder.build_object(dataset, **{
                "k": 3,
                "model_type": "sequence",
                "vector_size": 16
            })

        encoded_dataset = encoder.encode(dataset=dataset, params=config_params)

        self.assertIsNotNone(encoded_dataset.encoded_data)
        self.assertTrue(encoded_dataset.encoded_data.examples.shape[0] == 2)
        self.assertTrue(encoded_dataset.encoded_data.examples.shape[1] == 16)
        self.assertTrue(len(encoded_dataset.encoded_data.labels["T1D"]) == 2)
        self.assertTrue(encoded_dataset.encoded_data.labels["T1D"][0] == "T1D")
        self.assertTrue(isinstance(encoder, W2VRepertoireEncoder))

        shutil.rmtree(test_path)
