import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.EncoderParams import EncoderParams
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.DataEncoderParams import DataEncoderParams


class TestDataEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):
        path = EnvironmentSettings.root_path + "test/tmp/dataencoder/"
        PathBuilder.build(path)

        rep1 = Repertoire.build_from_sequence_objects([ReceptorSequence("AAA", identifier="1")],
                                                      metadata={"l1": 1, "l2": 2}, path=path)

        rep2 = Repertoire.build_from_sequence_objects([ReceptorSequence("ATA", identifier="2")],
                                                      metadata={"l1": 0, "l2": 3}, path=path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [0, 3])

        dataset = RepertoireDataset(repertoires=[rep1, rep2])
        encoder = Word2VecEncoder.build_object(dataset, **{
                    "k": 3,
                    "model_type": ModelType.SEQUENCE.name,
                    "vector_size": 6
                })

        res = DataEncoder.run(DataEncoderParams(
            dataset=dataset,
            encoder=encoder,
            encoder_params=EncoderParams(
                model={},
                pool_size=2,
                label_config=lc,
                result_path=path,
                filename="dataset.csv"
            )
        ))

        self.assertTrue(isinstance(res, RepertoireDataset))
        self.assertTrue(res.encoded_data.examples.shape[0] == 2)

        shutil.rmtree(path)
