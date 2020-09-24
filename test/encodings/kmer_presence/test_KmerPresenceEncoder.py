import os
import shutil
from unittest import TestCase

import numpy as np
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.encodings.kmer_presence.KmerPresenceEncoder import KmerPresenceEncoder
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder


class TestKmerPresenceEncoder(TestCase):
    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):
        path = EnvironmentSettings.root_path + "test/tmp/kmerfreqenc/"

        PathBuilder.build(path)
        rep1 = Repertoire.build_from_sequence_objects([ReceptorSequence("AAA", identifier="1"),
                                                       ReceptorSequence("ATA", identifier="2"),
                                                       ReceptorSequence("ATA", identifier='3')],
                                                      metadata={"l1": 1, "l2": 2, "subject_id": "1"}, path=path)

        rep2 = Repertoire.build_from_sequence_objects([ReceptorSequence("ATA", identifier="1"),
                                                       ReceptorSequence("TAA", identifier="2"),
                                                       ReceptorSequence("AAC", identifier="3")],
                                                      metadata={"l1": 0, "l2": 3, "subject_id": "2"}, path=path)

        rep3 = Repertoire.build_from_sequence_objects([ReceptorSequence("ATA", identifier="9"),
                                                       ReceptorSequence("TAA", identifier="8"),
                                                       ReceptorSequence("TAA", identifier="7"),
                                                       ReceptorSequence("CATASS", identifier="10")],
                                                      metadata={"l1": 2, "l2": 3, "subject_id": "2"}, path=path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 0, 2])
        lc.add_label("l2", [2, 3, 3])
        dataset = RepertoireDataset(repertoires=[rep1, rep2, rep3])

        encoder = KmerPresenceEncoder.build_object(dataset, **{"sequence_encoding": SequenceEncodingType.IDENTITY.name,
            "k": 3})

        d1 = encoder.encode(dataset, EncoderParams(
            result_path=path + "1/",
            label_config=lc,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        encoder = KmerPresenceEncoder.build_object(dataset, **{
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
            "k": 3
        })

        d2 = encoder.encode(dataset, EncoderParams(
            result_path=path + "2/",
            label_config=lc,
            pool_size=2,
            learn_model=True,
            model={},
            filename="dataset.csv"
        ))
        #

        # encoder = KmerPresenceEncoder.build_object(dataset, **{
        #     "sequence_encoding": SequenceEncodingType.GAPPED_KMER.name,
        #     "k": 3, "min_gap": 1, "max_gap": 3
        # })
        #
        # d3 = encoder.encode(dataset, EncoderParams(
        #     result_path=path + "2/",
        #     label_config=lc,
        #     pool_size=2,
        #     learn_model=True,
        #     model={},
        #     filename="dataset.csv"
        # ))

        shutil.rmtree(path)

        self.assertTrue(isinstance(d1, RepertoireDataset))
        self.assertTrue(isinstance(d2, RepertoireDataset))
        # print(d3.encoded_data.feature_names, d3.encoded_data.examples)
        # print(d3.encoded_data.examples[0, 1])
        self.assertEqual(1, np.round(d2.encoded_data.examples[2, 2], 2))
        self.assertEqual(0, np.round(d2.encoded_data.examples[1, 0], 2))
        self.assertTrue(isinstance(encoder, KmerPresenceEncoder))