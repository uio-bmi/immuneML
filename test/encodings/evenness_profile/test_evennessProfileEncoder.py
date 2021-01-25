import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.evenness_profile.EvennessProfileEncoder import EvennessProfileEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestEvennessEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):
        path = EnvironmentSettings.root_path / "test/tmp/evennessenc/"

        PathBuilder.build(path)

        rep1 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence("AAA", metadata=SequenceMetadata(count=10)) for i in range(1000)] +
                             [ReceptorSequence("AAA", metadata=SequenceMetadata(count=100)) for i in range(1000)] +
                             [ReceptorSequence("AAA", metadata=SequenceMetadata(count=1)) for i in range(1000)],
            metadata={"l1": "test_1", "l2": 2}, path=path)

        rep2 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence("AAA", metadata=SequenceMetadata(count=10)) for i in range(1000)],
            metadata={"l1": "test_2", "l2": 3}, path=path)

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
