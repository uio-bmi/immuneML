import shutil
from unittest import TestCase

import numpy as np

from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder


class TestKmerFrequencyEncoder(TestCase):
    def test_encode(self):

        path = EnvironmentSettings.root_path + "test/tmp/kmerfreqenc/"

        PathBuilder.build(path)

        rep1 = SequenceRepertoire.build_from_sequence_objects([ReceptorSequence("AAA", identifier="1"),
                                                               ReceptorSequence("ATA", identifier="2"),
                                                               ReceptorSequence("ATA", identifier='3')],
                                                              metadata={"l1": 1, "l2": 2}, path=path, identifier="1")

        rep2 = SequenceRepertoire.build_from_sequence_objects([ReceptorSequence("ATA", identifier="1"),
                                                               ReceptorSequence("TAA", identifier="2"),
                                                               ReceptorSequence("AAC", identifier="3")],
                                                              metadata={"l1": 0, "l2": 3}, path=path, identifier="2")

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [0, 3])

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        encoder = KmerFrequencyEncoder.create_encoder(dataset, {
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY,
                "reads": ReadsType.UNIQUE,
                "sequence_encoding": SequenceEncodingType.IDENTITY,
                "k": 3
            })

        d1 = encoder.encode(dataset, EncoderParams(
            result_path=path + "1/",
            label_configuration=lc,
            batch_size=2,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        encoder = KmerFrequencyEncoder.create_encoder(dataset, {
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY,
                "reads": ReadsType.UNIQUE,
                "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER,
                "k": 3
            })

        d2 = encoder.encode(dataset, EncoderParams(
            result_path=path + "2/",
            label_configuration=lc,
            batch_size=2,
            learn_model=True,
            model={},
            filename="dataset.csv"
        ))

        shutil.rmtree(path)

        self.assertTrue(isinstance(d1, RepertoireDataset))
        self.assertTrue(isinstance(d2, RepertoireDataset))
        self.assertEqual(0.67, np.round(d2.encoded_data.examples[0, 2], 2))
        self.assertTrue(isinstance(encoder, KmerFrequencyEncoder))
