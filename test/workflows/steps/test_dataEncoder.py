import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from immuneML.encodings.word2vec.model_creator.ModelType import ModelType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams


class TestDataEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):
        path = EnvironmentSettings.tmp_test_path / "data_encoder/"
        PathBuilder.build(path)

        rep1 = Repertoire.build_from_sequences([ReceptorSequence(sequence_aa="AAA", sequence_id="1")],
                                               metadata={"l1": 1, "l2": 2}, result_path=path)

        rep2 = Repertoire.build_from_sequences([ReceptorSequence(sequence_aa="ATA", sequence_id="2")],
                                               metadata={"l1": 0, "l2": 3}, result_path=path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [0, 3])

        dataset = RepertoireDataset(repertoires=[rep1, rep2])
        encoder = Word2VecEncoder.build_object(dataset, **{
            "k": 3,
            "model_type": ModelType.SEQUENCE.name,
            "vector_size": 6,
            "epochs": 10,
            "window": 5
        })

        res = DataEncoder.run(DataEncoderParams(
            dataset=dataset,
            encoder=encoder,
            encoder_params=EncoderParams(
                model={},
                pool_size=2,
                label_config=lc,
                result_path=path,
                sequence_type=SequenceType.AMINO_ACID,
                region_type=RegionType.IMGT_CDR3,
            )
        ))

        self.assertTrue(isinstance(res, RepertoireDataset))
        self.assertTrue(res.encoded_data.examples.shape[0] == 2)

        shutil.rmtree(path)
