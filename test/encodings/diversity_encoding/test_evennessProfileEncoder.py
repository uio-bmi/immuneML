import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.diversity_encoding.EvennessProfileEncoder import EvennessProfileEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestEvennessEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "evennessenc/")

        rep1 = Repertoire.build_from_sequences(
            sequences=[ReceptorSequence(sequence_aa="AAA", duplicate_count=10, vj_in_frame='T') for i in range(1000)] +
                      [ReceptorSequence(sequence_aa="AAA", duplicate_count=100, vj_in_frame='T') for i in range(1000)] +
                      [ReceptorSequence(sequence_aa="AAA", duplicate_count=1, vj_in_frame='T') for i in range(1000)],
            metadata={"l1": "test_1", "l2": 2}, result_path=path)

        rep2 = Repertoire.build_from_sequences(
            sequences=[ReceptorSequence(sequence_aa="AAA", duplicate_count=10) for i in range(1000)],
            metadata={"l1": "test_2", "l2": 3}, result_path=path)

        lc = LabelConfiguration()
        lc.add_label("l1", ["test_1", "test_2"])
        lc.add_label("l2", [0, 3])

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        encoder = EvennessProfileEncoder.build_object(dataset, **{
            "min_alpha": 0,
            "max_alpha": 10,
            "dimension": 51
        })

        d1 = encoder.encode(dataset, EncoderParams(
            result_path=path / "1/",
            label_config=lc,
        ))

        encoder = EvennessProfileEncoder.build_object(dataset, **{
            "min_alpha": 0,
            "max_alpha": 10,
            "dimension": 11
        })

        d2 = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=lc,
            pool_size=2
        ))

        self.assertAlmostEqual(d1.encoded_data.examples[0, 0], 1)
        self.assertAlmostEqual(d1.encoded_data.examples[0, 1], 0.786444)
        self.assertAlmostEqual(d1.encoded_data.examples[1, 0], 1)
        self.assertAlmostEqual(d1.encoded_data.examples[1, 1], 1)

        shutil.rmtree(path)
